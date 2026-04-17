#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

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
static constexpr int kMaxDepth = 40;
static constexpr int kQuietDepth = 4;
static constexpr int kKillerSlots = 2;
static constexpr float kSearchProbThreshold = 1.0f / 3.0f;
static constexpr float kRatFindPoints = 4.0f;
static constexpr float kRatMissPoints = 2.0f;
static constexpr float kInfinityScore = 1.0e9f;

struct TTEntry {
    int depth;
    float value;
    EncodedMove best_move;
    int node_type;  // 0 = exact, 1 = lower bound, 2 = upper bound
};

struct BoardKey {
    uint64_t primed_mask;
    uint64_t carpet_mask;
    uint64_t blocked_mask;
    uint8_t px;
    uint8_t py;
    uint8_t ox;
    uint8_t oy;
    int16_t p_points;
    int16_t o_points;
    int8_t p_turns;
    int8_t o_turns;

    bool operator==(const BoardKey& other) const {
        return primed_mask == other.primed_mask &&
               carpet_mask == other.carpet_mask &&
               blocked_mask == other.blocked_mask &&
               px == other.px &&
               py == other.py &&
               ox == other.ox &&
               oy == other.oy &&
               p_points == other.p_points &&
               o_points == other.o_points &&
               p_turns == other.p_turns &&
               o_turns == other.o_turns;
    }
};

struct BoardKeyHash {
    std::size_t operator()(const BoardKey& key) const {
        // Mix the compact board fields into one hash for the native TT.
        std::size_t h = std::hash<uint64_t>{}(key.primed_mask);
        h ^= std::hash<uint64_t>{}(key.carpet_mask) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.blocked_mask) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(
            uint32_t(key.px) | (uint32_t(key.py) << 8) |
            (uint32_t(key.ox) << 16) | (uint32_t(key.oy) << 24)
        ) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(
            uint32_t(uint16_t(key.p_points)) | (uint32_t(uint16_t(key.o_points)) << 16)
        ) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint16_t>{}(
            uint16_t(uint8_t(key.p_turns)) | (uint16_t(uint8_t(key.o_turns)) << 8)
        ) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct SearchTables {
    // Native transposition table: search memory for already-seen boards.
    std::unordered_map<BoardKey, TTEntry, BoardKeyHash> tt;

    // Killer moves: two strong cutoff-causing moves remembered per depth.
    EncodedMove killers[kMaxDepth + 1][kKillerSlots]{};
    bool has_killer[kMaxDepth + 1][kKillerSlots]{};

    // History heuristic: how often a move caused a cutoff before.
    int history[4][8][4]{};
};

struct SearchContext {
    DeadlineState clock;
    SearchTables tables;
};

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

static BoardKey make_board_key(const SearchState& state) {
    return BoardKey{
        state.primed_mask,
        state.carpet_mask,
        state.blocked_mask,
        uint8_t(state.px), uint8_t(state.py),
        uint8_t(state.ox), uint8_t(state.oy),
        int16_t(state.p_points), int16_t(state.o_points),
        int8_t(state.p_turns), int8_t(state.o_turns),
    };
}

static bool same_move(const EncodedMove& lhs, const EncodedMove& rhs) {
    return lhs.move_type == rhs.move_type &&
           lhs.direction == rhs.direction &&
           lhs.roll_length == rhs.roll_length;
}

static int move_roll_slot(const EncodedMove& move) {
    return move.move_type == kCarpetMoveType ? std::min(move.roll_length, 7) : 0;
}

static int history_score(const SearchTables& tables, const EncodedMove& move) {
    return tables.history[move.move_type][move_roll_slot(move)][move.direction];
}

static int killer_rank(const SearchTables& tables, int depth, const EncodedMove& move) {
    if (depth < 0 || depth > kMaxDepth) {
        return kKillerSlots;
    }
    for (int slot = 0; slot < kKillerSlots; ++slot) {
        if (tables.has_killer[depth][slot] && same_move(tables.killers[depth][slot], move)) {
            return slot;
        }
    }
    return kKillerSlots;
}

static void add_killer(SearchTables& tables, int depth, const EncodedMove& move) {
    // Remember moves that caused a cutoff, so we try them early later.
    if (depth < 0 || depth > kMaxDepth) {
        return;
    }
    if (tables.has_killer[depth][0] && same_move(tables.killers[depth][0], move)) {
        return;
    }
    if (!tables.has_killer[depth][0]) {
        tables.killers[depth][0] = move;
        tables.has_killer[depth][0] = true;
        return;
    }
    tables.killers[depth][1] = tables.killers[depth][0];
    tables.has_killer[depth][1] = tables.has_killer[depth][0];
    tables.killers[depth][0] = move;
    tables.has_killer[depth][0] = true;
}

static void add_history(SearchTables& tables, const EncodedMove& move, int depth) {
    // Reward moves that keep causing cutoffs. Deeper cutoffs matter more.
    tables.history[move.move_type][move_roll_slot(move)][move.direction] += depth * depth;
}

static bool tt_lookup(
    SearchTables& tables,
    const BoardKey& key,
    int depth,
    float alpha,
    float beta,
    float& out_value,
    EncodedMove* out_hint,
    bool* out_has_hint
) {
    if (out_has_hint != nullptr) {
        *out_has_hint = false;
    }

    auto it = tables.tt.find(key);
    if (it == tables.tt.end()) {
        return false;
    }

    const TTEntry& entry = it->second;
    if (out_hint != nullptr) {
        *out_hint = entry.best_move;
    }
    if (out_has_hint != nullptr) {
        *out_has_hint = true;
    }

    if (entry.depth < depth) {
        return false;
    }
    if (entry.node_type == 0) {
        out_value = entry.value;
        return true;
    }
    if (entry.node_type == 1 && entry.value >= beta) {
        out_value = entry.value;
        return true;
    }
    if (entry.node_type == 2 && entry.value <= alpha) {
        out_value = entry.value;
        return true;
    }
    return false;
}

static void tt_store(
    SearchTables& tables,
    const BoardKey& key,
    int depth,
    float value,
    const EncodedMove& best_move,
    float orig_alpha,
    float beta
) {
    int node_type = 0;
    if (value <= orig_alpha) {
        node_type = 2;
    } else if (value >= beta) {
        node_type = 1;
    }

    auto it = tables.tt.find(key);
    if (it == tables.tt.end() || depth >= it->second.depth) {
        tables.tt[key] = TTEntry{depth, value, best_move, node_type};
    }
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

static float threatened_run_value(const SearchState& state) {
    // If the opponent is near the far end of our best run, board play gets more urgent.
    float best_threat = 0.0f;
    uint64_t workers_mask = cell_bit(state.px, state.py) | cell_bit(state.ox, state.oy);

    for (int direction = 0; direction < 4; ++direction) {
        int run = 0;
        int x = state.px;
        int y = state.py;
        int far_x = state.px;
        int far_y = state.py;

        while (run < 7) {
            int nx = x + kStepX[direction];
            int ny = y + kStepY[direction];
            if (!inside_board(nx, ny)) {
                break;
            }

            uint64_t bit = cell_bit(nx, ny);
            if ((state.primed_mask & bit) == 0 || (workers_mask & bit) != 0) {
                break;
            }

            ++run;
            far_x = nx;
            far_y = ny;
            x = nx;
            y = ny;
        }

        if (run < 2) {
            continue;
        }

        int dx = state.ox > far_x ? state.ox - far_x : far_x - state.ox;
        int dy = state.oy > far_y ? state.oy - far_y : far_y - state.oy;
        if (dx + dy <= 1) {
            best_threat = std::max(best_threat, float(kCarpetPoints[run]));
        }
    }

    return best_threat;
}

static float search_barrier(const SearchState& state) {
    // Rat search should beat the board value we are giving up.
    float my_carpet = float(best_carpet_value(state, state.px, state.py, state.ox, state.oy));
    float threshold = (my_carpet < 1.0f) ? 0.5f : my_carpet;

    float steal_threat = threatened_run_value(state);
    if (steal_threat > 0.0f) {
        threshold += 0.5f * steal_threat;
    }

    if (state.p_turns <= 5) {
        threshold = std::max(0.5f, threshold - 1.0f);
    }
    return threshold;
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

static void sort_search_candidates(
    std::vector<EncodedMove>& moves,
    const SearchTables& tables,
    int depth,
    const EncodedMove* tt_hint
) {
    std::sort(moves.begin(), moves.end(), [&](const EncodedMove& lhs, const EncodedMove& rhs) {
        // TT move first: if the TT already liked a move here, try it first.
        bool lhs_tt = tt_hint != nullptr && same_move(lhs, *tt_hint);
        bool rhs_tt = tt_hint != nullptr && same_move(rhs, *tt_hint);
        if (lhs_tt != rhs_tt) {
            return lhs_tt;
        }

        // Killer moves next: these caused cutoffs before at this same depth.
        int lhs_killer = killer_rank(tables, depth, lhs);
        int rhs_killer = killer_rank(tables, depth, rhs);
        if (lhs_killer != rhs_killer) {
            return lhs_killer < rhs_killer;
        }

        int lhs_bucket = candidate_bucket(lhs);
        int rhs_bucket = candidate_bucket(rhs);
        if (lhs_bucket != rhs_bucket) {
            return lhs_bucket < rhs_bucket;
        }

        if (lhs.move_type == kCarpetMoveType && rhs.move_type == kCarpetMoveType) {
            // Match the Python path: longer carpets still matter most.
            if (lhs.roll_length != rhs.roll_length) {
                return lhs.roll_length > rhs.roll_length;
            }
        }

        // Within a move bucket, try moves that caused more cutoffs first.
        int lhs_hist = history_score(tables, lhs);
        int rhs_hist = history_score(tables, rhs);
        if (lhs_hist != rhs_hist) {
            return lhs_hist > rhs_hist;
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

static std::vector<EncodedMove> collect_search_moves(
    const SearchState& state,
    const SearchTables& tables,
    int depth,
    const EncodedMove* tt_hint
) {
    std::vector<EncodedMove> moves;
    append_step_moves(state, moves);
    append_carpet_moves(state, moves);
    sort_search_candidates(moves, tables, depth, tt_hint);
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
    float threshold = search_barrier(state);
    if (ev < threshold || ev <= 0.0f) {
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
    SearchContext& ctx
) {
    if (deadline_hit(ctx.clock)) {
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
        float score = -quiet_extension(child, -beta, -alpha, quiet_depth - 1, ctx);
        if (ctx.clock.timed_out) {
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
    SearchContext& ctx
) {
    if (deadline_hit(ctx.clock)) {
        return 0.0f;
    }

    if (depth <= 0 || state.p_turns <= 0) {
        return quiet_extension(state, alpha, beta, kQuietDepth, ctx);
    }

    BoardKey key = make_board_key(state);
    float tt_value = 0.0f;
    EncodedMove tt_hint{};
    bool has_tt_hint = false;
    if (tt_lookup(ctx.tables, key, depth, alpha, beta, tt_value, &tt_hint, &has_tt_hint)) {
        return tt_value;
    }

    auto moves = collect_search_moves(state, ctx.tables, depth, has_tt_hint ? &tt_hint : nullptr);
    if (moves.empty()) {
        return evaluate_position(state);
    }

    float best = -std::numeric_limits<float>::infinity();
    float orig_alpha = alpha;
    int searched_count = 0;
    EncodedMove best_move = moves.front();

    for (const auto& move : moves) {
        if (deadline_hit(ctx.clock)) {
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
            score = -search_branch(child, depth - 2, -(alpha + 1.0f), -alpha, ctx);
            if (!ctx.clock.timed_out && score > alpha) {
                score = -search_branch(child, depth - 1, -beta, -alpha, ctx);
            }
        } else {
            score = -search_branch(child, depth - 1, -beta, -alpha, ctx);
        }

        if (ctx.clock.timed_out) {
            return 0.0f;
        }
        if (score > best) {
            best = score;
            best_move = move;
        }
        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) {
            // A cutoff means this move is worth remembering globally.
            add_killer(ctx.tables, depth, move);
            add_history(ctx.tables, move, depth);
            break;
        }
        ++searched_count;
    }

    tt_store(ctx.tables, key, depth, best, best_move, orig_alpha, beta);
    return best;
}

static bool evaluate_root_choices(
    const SearchState& root,
    const std::vector<double>& belief,
    int depth,
    SearchContext& ctx,
    RootChoice& out
) {
    BoardKey key = make_board_key(root);
    float ignored = 0.0f;
    EncodedMove tt_hint{};
    bool has_tt_hint = false;
    tt_lookup(ctx.tables, key, depth, -kInfinityScore, kInfinityScore, ignored, &tt_hint, &has_tt_hint);

    auto moves = collect_search_moves(root, ctx.tables, depth, has_tt_hint ? &tt_hint : nullptr);

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

    for (const auto& move : moves) {
        if (deadline_hit(ctx.clock)) {
            return false;
        }
        SearchState child = advance_and_swap(root, move);
        float score = -search_branch(child, depth - 1, -beta, -alpha, ctx);
        if (ctx.clock.timed_out) {
            return false;
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

    return out.has_move;
}

}  // namespace yolanda_ref

static PyObject* search_entrypoint(PyObject* self, PyObject* args) {
    (void)self;

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
    yolanda_ref::SearchContext ctx{};
    ctx.clock.deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(safe_budget)
        );
    ctx.clock.timed_out = false;

    yolanda_ref::RootChoice best{
        false,
        {yolanda_ref::kPlainMoveType, 0, 0},
        -yolanda_ref::kInfinityScore
    };
    for (int depth = 1; depth <= yolanda_ref::kMaxDepth; ++depth) {
        yolanda_ref::RootChoice current{};
        if (!yolanda_ref::evaluate_root_choices(root, belief, depth, ctx, current)) {
            break;
        }
        best = current;
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

static PyMethodDef YolandaSearchMethods[] = {
    {
        "search",
        search_entrypoint,
        METH_VARARGS,
        "Native root search entrypoint for Yolanda."
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
