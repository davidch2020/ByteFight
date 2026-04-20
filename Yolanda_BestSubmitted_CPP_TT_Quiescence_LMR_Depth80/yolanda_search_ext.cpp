#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>   // [TT] memset for TT clear
#include <limits>
#include <random>    // [TT] mt19937_64 for Zobrist key seeding
#include <vector>

// =============================================================================
// HIGH-LEVEL OVERVIEW (where the changes are)
// -----------------------------------------------------------------------------
// This file is the same refactored search as before, with a Transposition Table
// and Zobrist hashing added.  Every new / modified section is marked with [TT].
//
// What a TT does:
//   1. Before searching a position, we hash the board to a 64-bit key and look
//      up a cached result.  If the cached result was computed at >= our current
//      depth and the alpha/beta bounds match, we return it immediately.  This
//      prunes huge subtrees — the same positions often appear via different
//      move orders ("transpositions").
//   2. Even when the cached value is unusable (shallower depth), the entry
//      still tells us the best move found last time.  We try that move first,
//      which produces beta-cutoffs earlier and improves alpha-beta pruning.
//
// The Python side already had a TT.  This version puts it in the C++ path —
// previously the C++ backend had no TT at all, which wasted a lot of work.
//
// New pieces added below:
//   - Zobrist key arrays + `ensure_zobrist_ready()`   (seeded once per process)
//   - `compute_hash()`                                (board state -> uint64)
//   - `TTEntry` + fixed-size `g_tt` table             (1M entries, ~32 MB)
//   - `tt_probe()` / `tt_store()`                     (depth-preferred replace)
//   - TT probe/store + TT-hint move ordering in `search_branch` and
//     `evaluate_root_choices`
//   - `ensure_zobrist_ready()` called from `search_entrypoint`
// =============================================================================

namespace yolanda_ref {

// Keep the encoded move compact so the Python bridge stays simple.
struct EncodedMove {
    int move_type;
    int direction;
    int roll_length;
};

// SearchState stores the board from the current side-to-move perspective.
struct SearchState {
    uint64_t primed_mask;
    uint64_t carpet_mask;
    uint64_t blocked_mask;
    int px;
    int py;
    int ox;
    int oy;
    int p_points;
    int o_points;
    int p_turns;
    int o_turns;
};

// RootChoice keeps the best fully searched root answer so far.
struct RootChoice {
    bool has_move;
    EncodedMove move;
    float score;
};

// DeadlineState is shared by every recursive call.
struct DeadlineState {
    std::chrono::steady_clock::time_point deadline;
    bool timed_out;
};

static constexpr int kBoardSize = 8;
static constexpr int kPlainMoveType = 0;
static constexpr int kPrimeMoveType = 1;
static constexpr int kCarpetMoveType = 2;
static constexpr int kSearchMoveType = 3;
static constexpr int kStepX[4] = {0, 1, 0, -1};
static constexpr int kStepY[4] = {-1, 0, 1, 0};
static constexpr int kCarpetPoints[8] = {0, -1, 2, 4, 6, 10, 15, 21};
static constexpr int kMaxDepth = 80;
static constexpr int kQuietDepth = 4;
static constexpr float kSearchProbThreshold = 0.5f;
static constexpr float kRatFindPoints = 4.0f;
static constexpr float kRatMissPoints = 2.0f;
static constexpr float kInfinityScore = 1.0e9f;

// =============================================================================
// [TT] Transposition Table + Zobrist hashing
// =============================================================================

// One 64-bit random value per (feature, cell).  XOR-ing the values for every
// feature present on the board gives a well-distributed 64-bit fingerprint of
// the state.  These arrays are filled once at process start.
static uint64_t zobrist_primed[64];
static uint64_t zobrist_carpet[64];
static uint64_t zobrist_ppos[64];
static uint64_t zobrist_opos[64];
static bool zobrist_initialized = false;

// Fill the Zobrist tables once per process.  A deterministic seed means two
// runs produce the same keys, which makes TT behavior reproducible.
static void ensure_zobrist_ready() {
    if (zobrist_initialized) return;
    std::mt19937_64 rng(0xD3ADB33FULL);
    for (int i = 0; i < 64; ++i) {
        zobrist_primed[i] = rng();
        zobrist_carpet[i] = rng();
        zobrist_ppos[i]   = rng();
        zobrist_opos[i]   = rng();
    }
    zobrist_initialized = true;
}

// Full hash recompute from a SearchState.  Simple, safe, ~15-30 ns per call.
// Can be upgraded to an incremental XOR-in/XOR-out scheme later if needed.
static uint64_t compute_hash(const SearchState& s) {
    uint64_t h = 0;

    // XOR in each set primed cell.
    uint64_t p = s.primed_mask;
    while (p) {
        int bit = __builtin_ctzll(p);
        h ^= zobrist_primed[bit];
        p &= p - 1;
    }
    // XOR in each set carpet cell.
    uint64_t c = s.carpet_mask;
    while (c) {
        int bit = __builtin_ctzll(c);
        h ^= zobrist_carpet[bit];
        c &= c - 1;
    }
    // Worker positions.  p_pos and o_pos live in separate arrays so "me at A,
    // opp at B" hashes differently from "me at B, opp at A".
    h ^= zobrist_ppos[s.py * 8 + s.px];
    h ^= zobrist_opos[s.oy * 8 + s.ox];

    // Scores and remaining turns affect the evaluation, so they must be part
    // of the key — two otherwise-identical boards with different scores are
    // different search states.  We use simple large-prime multipliers to mix.
    h ^= uint64_t(s.p_points) * 0x9E3779B97F4A7C15ULL;
    h ^= uint64_t(s.o_points) * 0xBF58476D1CE4E5B9ULL;
    h ^= uint64_t(s.p_turns)  * 0x94D049BB133111EBULL;
    h ^= uint64_t(s.o_turns)  * 0xD1B54A32D192ED03ULL;
    return h;
}

// Bound classification for the stored value.
//   EXACT = the true minimax score
//   LOWER = a lower bound (fail-high: true score >= value)
//   UPPER = an upper bound (fail-low:  true score <= value)
static constexpr uint8_t kTTExact = 0;
static constexpr uint8_t kTTLower = 1;
static constexpr uint8_t kTTUpper = 2;

// A TT slot.  Laid out small so we pack cache lines reasonably.
struct TTEntry {
    uint64_t    key;    // Zobrist hash.  key == 0 is the "empty slot" sentinel.
    float       value;  // Stored score from the side-to-move's perspective.
    EncodedMove best;   // Best move found at this position (for ordering).
    int16_t     depth;  // Search depth used to produce `value`.
    uint8_t     flag;   // EXACT / LOWER / UPPER.
    uint8_t     _pad;
};

// Fixed-size table.  Power-of-two size lets us index with a bit mask
// instead of %.  No heap allocation during search.
static constexpr size_t kTTSize = 1 << 20;   // 1M entries
static constexpr size_t kTTMask = kTTSize - 1;
static TTEntry g_tt[kTTSize];                // zero-initialized by the loader

// Instrumentation counters so the Python side can report TT activity.
// These do not affect search behavior.
static uint64_t g_tt_probes = 0;
static uint64_t g_tt_hits = 0;
static uint64_t g_tt_stores = 0;
static int g_last_completed_depth = 0;

// Probe: return the entry if the key matches, else null.
static inline const TTEntry* tt_probe(uint64_t key) {
    ++g_tt_probes;
    const TTEntry& e = g_tt[key & kTTMask];
    if (e.key == key) {
        ++g_tt_hits;
        return &e;
    }
    return nullptr;
}

// Store: depth-preferred replacement.  We only overwrite an existing slot if
// the new result was searched at least as deep — this protects expensive
// deep-search results from being clobbered by cheap shallow probes.
static inline void tt_store(uint64_t key, int depth, float value,
                            uint8_t flag, EncodedMove best) {
    TTEntry& e = g_tt[key & kTTMask];
    if (e.key != key || e.depth <= depth) {
        ++g_tt_stores;
        e.key   = key;
        e.depth = static_cast<int16_t>(depth);
        e.flag  = flag;
        e.best  = best;
        e.value = value;
    }
}

// Optional: wipe the whole TT (not called automatically — kept in case we want
// to flush between games).  The zero-init of g_tt already gives a clean start
// at process load.
static inline void tt_clear_all() {
    std::memset(g_tt, 0, sizeof(g_tt));
}

static inline void tt_reset_counters() {
    g_tt_probes = 0;
    g_tt_hits = 0;
    g_tt_stores = 0;
    g_last_completed_depth = 0;
}

// =============================================================================
// End [TT] additions — everything below is the same search logic, with small
// integration hooks marked [TT] where they call into the table above.
// =============================================================================

static inline uint64_t cell_bit(int x, int y) {
    return 1ULL << (y * kBoardSize + x);
}

static inline bool inside_board(int x, int y) {
    return 0 <= x && x < kBoardSize && 0 <= y && y < kBoardSize;
}

static inline bool bit_is_set(uint64_t mask, int x, int y) {
    return (mask & cell_bit(x, y)) != 0;
}

static inline bool deadline_hit(DeadlineState& clock_state) {
    if (clock_state.timed_out) {
        return true;
    }
    if (std::chrono::steady_clock::now() >= clock_state.deadline) {
        clock_state.timed_out = true;
        return true;
    }
    return false;
}

static int best_carpet_value(
    const SearchState& state,
    int start_x,
    int start_y,
    int block_x,
    int block_y
) {
    int best = 0;
    for (int direction = 0; direction < 4; ++direction) {
        int x = start_x + kStepX[direction];
        int y = start_y + kStepY[direction];
        int run = 0;
        while (inside_board(x, y)) {
            if (x == block_x && y == block_y) {
                break;
            }
            if (!bit_is_set(state.primed_mask, x, y)) {
                break;
            }
            ++run;
            x += kStepX[direction];
            y += kStepY[direction];
        }
        if (run >= 1) {
            best = std::max(best, kCarpetPoints[std::min(run, 7)]);
        }
    }
    return best;
}

static int adjacent_prime_count(const SearchState& state, int x, int y) {
    int adjacent = 0;
    for (int direction = 0; direction < 4; ++direction) {
        int nx = x + kStepX[direction];
        int ny = y + kStepY[direction];
        if (inside_board(nx, ny) && bit_is_set(state.primed_mask, nx, ny)) {
            ++adjacent;
        }
    }
    return adjacent;
}

static float evaluate_position(const SearchState& state) {
    // Keep the same lightweight eval; only names and structure changed.
    int turns = std::max(1, state.p_turns);
    float score = float(state.p_points - state.o_points) * (1.0f + 0.12f * float(turns));
    score += 0.5f * float(best_carpet_value(state, state.px, state.py, state.ox, state.oy));
    score -= 0.3f * float(best_carpet_value(state, state.ox, state.oy, state.px, state.py));
    score += 0.12f * float(__builtin_popcountll(state.primed_mask));

    if (((state.primed_mask | state.carpet_mask) & cell_bit(state.px, state.py)) == 0) {
        score += 0.4f;
    }

    score += 0.2f * float(adjacent_prime_count(state, state.px, state.py));
    return score;
}

static int candidate_bucket(const EncodedMove& move) {
    if (move.move_type == kCarpetMoveType) {
        return 0;
    }
    if (move.move_type == kPrimeMoveType) {
        return 1;
    }
    return 2;
}

static void append_step_moves(const SearchState& state, std::vector<EncodedMove>& out) {
    uint64_t workers_mask = cell_bit(state.px, state.py) | cell_bit(state.ox, state.oy);
    uint64_t blocked_cells = state.blocked_mask | state.primed_mask | workers_mask;
    bool can_prime = ((state.primed_mask | state.carpet_mask) & cell_bit(state.px, state.py)) == 0;

    for (int direction = 0; direction < 4; ++direction) {
        int nx = state.px + kStepX[direction];
        int ny = state.py + kStepY[direction];
        if (!inside_board(nx, ny)) {
            continue;
        }

        uint64_t next_bit = cell_bit(nx, ny);
        if ((blocked_cells & next_bit) == 0) {
            out.push_back({kPlainMoveType, direction, 0});
            if (can_prime) {
                out.push_back({kPrimeMoveType, direction, 0});
            }
        }
    }
}

static void append_carpet_moves(const SearchState& state, std::vector<EncodedMove>& out) {
    for (int direction = 0; direction < 4; ++direction) {
        int x = state.px;
        int y = state.py;
        for (int roll = 1; roll < kBoardSize; ++roll) {
            x += kStepX[direction];
            y += kStepY[direction];
            if (!inside_board(x, y)) {
                break;
            }
            if (x == state.ox && y == state.oy) {
                break;
            }
            if (!bit_is_set(state.primed_mask, x, y)) {
                break;
            }
            out.push_back({kCarpetMoveType, direction, roll});
        }
    }
}

static void sort_candidates(std::vector<EncodedMove>& moves) {
    std::sort(moves.begin(), moves.end(), [](const EncodedMove& lhs, const EncodedMove& rhs) {
        int lhs_bucket = candidate_bucket(lhs);
        int rhs_bucket = candidate_bucket(rhs);
        if (lhs_bucket != rhs_bucket) {
            return lhs_bucket < rhs_bucket;
        }
        if (lhs.move_type == kCarpetMoveType &&
            rhs.move_type == kCarpetMoveType &&
            lhs.roll_length != rhs.roll_length) {
            return lhs.roll_length > rhs.roll_length;
        }
        if (lhs.direction != rhs.direction) {
            return lhs.direction < rhs.direction;
        }
        return lhs.roll_length > rhs.roll_length;
    });
}

static std::vector<EncodedMove> collect_moves(const SearchState& state) {
    std::vector<EncodedMove> moves;
    append_step_moves(state, moves);
    append_carpet_moves(state, moves);
    sort_candidates(moves);
    return moves;
}

static SearchState advance_and_swap(const SearchState& state, const EncodedMove& move) {
    SearchState next = state;

    if (move.move_type == kPlainMoveType) {
        next.px += kStepX[move.direction];
        next.py += kStepY[move.direction];
    } else if (move.move_type == kPrimeMoveType) {
        next.primed_mask |= cell_bit(next.px, next.py);
        next.px += kStepX[move.direction];
        next.py += kStepY[move.direction];
        next.p_points += 1;
    } else if (move.move_type == kCarpetMoveType) {
        int x = next.px;
        int y = next.py;
        for (int step = 0; step < move.roll_length; ++step) {
            x += kStepX[move.direction];
            y += kStepY[move.direction];
            uint64_t bit = cell_bit(x, y);
            next.primed_mask &= ~bit;
            next.carpet_mask |= bit;
        }
        next.px = x;
        next.py = y;
        next.p_points += kCarpetPoints[std::min(move.roll_length, 7)];
    }

    next.p_turns -= 1;

    // Keep negamax simple by flipping to the next player-to-move view.
    std::swap(next.px, next.ox);
    std::swap(next.py, next.oy);
    std::swap(next.p_points, next.o_points);
    std::swap(next.p_turns, next.o_turns);
    return next;
}

static bool decode_belief_sequence(PyObject* belief_obj, std::vector<double>& belief) {
    PyObject* seq = PySequence_Fast(belief_obj, "belief must be a sequence");
    if (seq == nullptr) {
        return false;
    }

    Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    belief.resize(static_cast<size_t>(count));
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < count; ++i) {
        belief[static_cast<size_t>(i)] = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
    }

    Py_DECREF(seq);
    return true;
}

static bool select_search_option(
    const SearchState& state,
    const std::vector<double>& belief,
    EncodedMove& search_move,
    float& search_score
) {
    if (belief.empty()) {
        return false;
    }

    auto it = std::max_element(belief.begin(), belief.end());
    double probability = *it;
    if (probability < kSearchProbThreshold) {
        return false;
    }

    float ev = kRatFindPoints * static_cast<float>(probability) -
               kRatMissPoints * (1.0f - static_cast<float>(probability));
    int board_cost = best_carpet_value(state, state.px, state.py, state.ox, state.oy);
    if (board_cost > ev + 2.0f || ev <= 0.0f) {
        return false;
    }

    int index = static_cast<int>(std::distance(belief.begin(), it));
    search_move = {kSearchMoveType, index & 7, index >> 3};
    search_score = ev;
    return true;
}

static float quiet_extension(
    const SearchState& state,
    float alpha,
    float beta,
    int quiet_depth,
    DeadlineState& clock_state
) {
    if (deadline_hit(clock_state)) {
        return 0.0f;
    }

    float stand_pat = evaluate_position(state);
    if (stand_pat >= beta) {
        return beta;
    }
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }
    if (quiet_depth <= 0) {
        return alpha;
    }

    auto moves = collect_moves(state);
    for (const auto& move : moves) {
        // Extend only meaningful carpet tactics.
        if (move.move_type != kCarpetMoveType || move.roll_length < 2) {
            continue;
        }
        SearchState child = advance_and_swap(state, move);
        float score = -quiet_extension(child, -beta, -alpha, quiet_depth - 1, clock_state);
        if (clock_state.timed_out) {
            return 0.0f;
        }
        if (score >= beta) {
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }
    return alpha;
}

static float search_branch(
    const SearchState& state,
    int depth,
    float alpha,
    float beta,
    DeadlineState& clock_state
) {
    if (deadline_hit(clock_state)) {
        return 0.0f;
    }

    if (depth <= 0 || state.p_turns <= 0) {
        return quiet_extension(state, alpha, beta, kQuietDepth, clock_state);
    }

    // [TT] Save original alpha so we can classify the bound type on store.
    const float orig_alpha = alpha;

    // [TT] Probe the table.  Three possible outcomes:
    //   (a) exact or matching-bound cached value at >= our depth -> return it
    //   (b) entry exists but cannot prune -> use its best move for ordering
    //   (c) no entry -> nothing special; fall through
    const uint64_t hash = compute_hash(state);
    EncodedMove tt_move{-1, -1, -1};
    if (const TTEntry* entry = tt_probe(hash)) {
        tt_move = entry->best;
        if (entry->depth >= depth) {
            if (entry->flag == kTTExact)                                 return entry->value;
            if (entry->flag == kTTLower && entry->value >= beta)         return entry->value;
            if (entry->flag == kTTUpper && entry->value <= alpha)        return entry->value;
        }
    }

    auto moves = collect_moves(state);
    if (moves.empty()) {
        return evaluate_position(state);
    }

    // [TT] Move ordering: swap the TT-best move to the front if we have one.
    // This is often a bigger win than the early-return, because trying the
    // right move first lets alpha-beta cut off the rest.
    if (tt_move.move_type >= 0) {
        for (size_t i = 1; i < moves.size(); ++i) {
            if (moves[i].move_type   == tt_move.move_type &&
                moves[i].direction   == tt_move.direction &&
                moves[i].roll_length == tt_move.roll_length) {
                std::swap(moves[0], moves[i]);
                break;
            }
        }
    }

    float best = -std::numeric_limits<float>::infinity();
    EncodedMove best_move = moves[0];    // [TT] Remember which move produced `best` for storing in the TT.
    int searched_count = 0;

    for (const auto& move : moves) {
        if (deadline_hit(clock_state)) {
            return 0.0f;
        }

        SearchState child = advance_and_swap(state, move);
        float score = 0.0f;

        // Keep the same conservative LMR policy.
        bool use_lmr = (
            depth >= 3 &&
            searched_count >= 2 &&
            move.move_type == kPlainMoveType
        );
        if (use_lmr) {
            score = -search_branch(child, depth - 2, -(alpha + 1.0f), -alpha, clock_state);
            if (!clock_state.timed_out && score > alpha) {
                score = -search_branch(child, depth - 1, -beta, -alpha, clock_state);
            }
        } else {
            score = -search_branch(child, depth - 1, -beta, -alpha, clock_state);
        }

        if (clock_state.timed_out) {
            return 0.0f;
        }
        if (score > best) {
            best = score;
            best_move = move;               // [TT] Track best move.
        }
        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) {
            break;
        }
        ++searched_count;
    }

    // [TT] Classify the result and store it.
    //   best <= orig_alpha  -> we never exceeded the incoming alpha (fail-low)
    //   best >= beta        -> we caused a beta cutoff              (fail-high)
    //   else                -> exact minimax value
    uint8_t flag = kTTExact;
    if      (best <= orig_alpha) flag = kTTUpper;
    else if (best >= beta)       flag = kTTLower;
    tt_store(hash, depth, best, flag, best_move);

    return best;
}

static bool evaluate_root_choices(
    const SearchState& root,
    const std::vector<double>& belief,
    int depth,
    DeadlineState& clock_state,
    RootChoice& out
) {
    auto moves = collect_moves(root);

    // [TT] Use the TT's best move from prior iterative-deepening iterations
    // as the first move to try at the root.  This is usually the previous
    // iteration's best answer, which almost always stays best at the next
    // depth and produces an immediate high alpha.
    const uint64_t root_hash = compute_hash(root);
    EncodedMove tt_move{-1, -1, -1};
    if (const TTEntry* entry = tt_probe(root_hash)) {
        tt_move = entry->best;
    }
    if (tt_move.move_type >= 0) {
        for (size_t i = 0; i < moves.size(); ++i) {
            if (moves[i].move_type   == tt_move.move_type &&
                moves[i].direction   == tt_move.direction &&
                moves[i].roll_length == tt_move.roll_length) {
                std::swap(moves[0], moves[i]);
                break;
            }
        }
    }

    out.has_move = false;
    out.score = -kInfinityScore;
    float alpha = -kInfinityScore;
    float beta = kInfinityScore;

    // Rat search is treated as one root-level candidate alongside board moves.
    EncodedMove search_move{};
    float search_score = 0.0f;
    if (select_search_option(root, belief, search_move, search_score)) {
        out.has_move = true;
        out.move = search_move;
        out.score = search_score;
        alpha = search_score;
    }

    if (moves.empty()) {
        return out.has_move;
    }

    // [TT] Track the best *regular* (non-search) move separately so we can
    // still store a useful move-ordering hint even if a rat-search move wins.
    EncodedMove best_regular = moves[0];
    float best_regular_score = -kInfinityScore;
    bool have_regular = false;

    for (const auto& move : moves) {
        if (deadline_hit(clock_state)) {
            return false;
        }
        SearchState child = advance_and_swap(root, move);
        float score = -search_branch(child, depth - 1, -beta, -alpha, clock_state);
        if (clock_state.timed_out) {
            return false;
        }
        if (!have_regular || score > best_regular_score) {
            best_regular = move;
            best_regular_score = score;
            have_regular = true;
        }
        if (!out.has_move || score > out.score) {
            out.has_move = true;
            out.move = move;
            out.score = score;
        }
        if (score > alpha) {
            alpha = score;
        }
    }

    // [TT] Store the best regular move at the root so the next ID iteration
    // immediately picks it up as the first move to try.  We store the
    // regular-move score (not a rat-search score) because the TT entry is
    // about minimax-style board play, not the rat sub-game.
    if (have_regular) {
        tt_store(root_hash, depth, best_regular_score, kTTExact, best_regular);
    }

    return out.has_move;
}

}  // namespace yolanda_ref

static PyObject* search_entrypoint(PyObject* self, PyObject* args) {
    (void)self;

    // [TT] Make sure Zobrist keys are filled before any hashing happens.
    yolanda_ref::ensure_zobrist_ready();

    unsigned long long primed_mask;
    unsigned long long carpet_mask;
    unsigned long long blocked_mask;
    int px;
    int py;
    int p_points;
    int p_turns;
    int ox;
    int oy;
    int o_points;
    int o_turns;
    double budget;
    PyObject* belief_obj = nullptr;

    if (!PyArg_ParseTuple(
            args,
            "KKKiiiiiiiidO",
            &primed_mask, &carpet_mask, &blocked_mask,
            &px, &py, &p_points, &p_turns,
            &ox, &oy, &o_points, &o_turns,
            &budget, &belief_obj)) {
        return nullptr;
    }

    std::vector<double> belief;
    if (!yolanda_ref::decode_belief_sequence(belief_obj, belief)) {
        return nullptr;
    }

    yolanda_ref::SearchState root{
        primed_mask,
        carpet_mask,
        blocked_mask,
        px,
        py,
        ox,
        oy,
        p_points,
        o_points,
        p_turns,
        o_turns
    };

    double safe_budget = std::max(0.01, budget - 0.01);
    yolanda_ref::DeadlineState clock_state{};
    clock_state.deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(safe_budget)
        );
    clock_state.timed_out = false;

    yolanda_ref::RootChoice best{
        false,
        {yolanda_ref::kPlainMoveType, 0, 0},
        -yolanda_ref::kInfinityScore
    };
    for (int depth = 1; depth <= yolanda_ref::kMaxDepth; ++depth) {
        yolanda_ref::RootChoice current{};
        if (!yolanda_ref::evaluate_root_choices(root, belief, depth, clock_state, current)) {
            break;
        }
        best = current;
        yolanda_ref::g_last_completed_depth = depth;
    }

    if (!best.has_move) {
        Py_RETURN_NONE;
    }

    return Py_BuildValue(
        "(iiif)",
        best.move.move_type,
        best.move.direction,
        best.move.roll_length,
        best.score
    );
}

static PyObject* tt_stats_entrypoint(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    return Py_BuildValue(
        "(KKKi)",
        yolanda_ref::g_tt_hits,
        yolanda_ref::g_tt_probes,
        yolanda_ref::g_tt_stores,
        yolanda_ref::g_last_completed_depth
    );
}

static PyObject* reset_tt_counters_entrypoint(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    yolanda_ref::tt_reset_counters();
    Py_RETURN_NONE;
}

static PyMethodDef YolandaSearchMethods[] = {
    {
        "search",
        search_entrypoint,
        METH_VARARGS,
        "Native root search entrypoint for Yolanda."
    },
    {
        "tt_stats",
        tt_stats_entrypoint,
        METH_NOARGS,
        "Return TT (hits, probes, stores)."
    },
    {
        "reset_tt_counters",
        reset_tt_counters_entrypoint,
        METH_NOARGS,
        "Reset TT counters."
    },
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef yolanda_search_module = {
    PyModuleDef_HEAD_INIT,
    "yolanda_search_ext",
    "Optional native search backend for Yolanda.",
    -1,
    YolandaSearchMethods
};

PyMODINIT_FUNC PyInit_yolanda_search_ext(void) {
    return PyModule_Create(&yolanda_search_module);
}
