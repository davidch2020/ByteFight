#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <vector>

// Keep the native board compact so search loops stay cheap.
struct NativeMove {
    int move_type;
    int direction;
    int roll_length;
};

struct NativeBoard {
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

struct RootResult {
    bool has_move;
    NativeMove move;
    float score;
};

struct SearchContext {
    std::chrono::steady_clock::time_point deadline;
    bool timed_out;
};

static constexpr int BOARD_N = 8;
static constexpr int PLAIN_MT = 0;
static constexpr int PRIME_MT = 1;
static constexpr int CARPET_MT = 2;
static constexpr int SEARCH_MT = 3;
static constexpr int DX[4] = {0, 1, 0, -1};
static constexpr int DY[4] = {-1, 0, 1, 0};
static constexpr int CPT[8] = {0, -1, 2, 4, 6, 10, 15, 21};
static constexpr int MAX_DEPTH = 40;
static constexpr int QSEARCH_DEPTH = 4;
static constexpr float SEARCH_PROB_THRESHOLD = 0.5f;
static constexpr float RAT_FIND_PTS = 4.0f;
static constexpr float RAT_MISS_PTS = 2.0f;
static constexpr float INF = 1.0e9f;

static inline uint64_t bit_at(int x, int y) {
    return 1ULL << (y * BOARD_N + x);
}

static inline bool on_board(int x, int y) {
    return 0 <= x && x < BOARD_N && 0 <= y && y < BOARD_N;
}

static inline bool has_bit(uint64_t mask, int x, int y) {
    return (mask & bit_at(x, y)) != 0;
}

static inline bool out_of_time(SearchContext& ctx) {
    if (ctx.timed_out) {
        return true;
    }
    if (std::chrono::steady_clock::now() >= ctx.deadline) {
        ctx.timed_out = true;
        return true;
    }
    return false;
}

static int best_carpet_run(const NativeBoard& b, int sx, int sy, int tx, int ty) {
    int best = 0;
    for (int d = 0; d < 4; ++d) {
        int nx = sx + DX[d];
        int ny = sy + DY[d];
        int run = 0;
        while (on_board(nx, ny)) {
            if (nx == tx && ny == ty) {
                break;
            }
            if (!has_bit(b.primed_mask, nx, ny)) {
                break;
            }
            ++run;
            nx += DX[d];
            ny += DY[d];
        }
        if (run >= 1) {
            best = std::max(best, CPT[std::min(run, 7)]);
        }
    }
    return best;
}

static int adjacent_primed_count(const NativeBoard& b, int x, int y) {
    int adj = 0;
    for (int d = 0; d < 4; ++d) {
        int nx = x + DX[d];
        int ny = y + DY[d];
        if (on_board(nx, ny) && has_bit(b.primed_mask, nx, ny)) {
            ++adj;
        }
    }
    return adj;
}

static float eval_board(const NativeBoard& b) {
    // Match Yolanda's lightweight board heuristic in native code.
    int turns = std::max(1, b.p_turns);
    float score = float(b.p_points - b.o_points) * (1.0f + 0.12f * float(turns));
    score += 0.5f * float(best_carpet_run(b, b.px, b.py, b.ox, b.oy));
    score -= 0.3f * float(best_carpet_run(b, b.ox, b.oy, b.px, b.py));
    score += 0.12f * float(__builtin_popcountll(b.primed_mask));

    if (((b.primed_mask | b.carpet_mask) & bit_at(b.px, b.py)) == 0) {
        score += 0.4f;
    }

    score += 0.2f * float(adjacent_primed_count(b, b.px, b.py));
    return score;
}

static int move_sort_bucket(const NativeMove& m) {
    if (m.move_type == CARPET_MT) {
        return 0;
    }
    if (m.move_type == PRIME_MT) {
        return 1;
    }
    return 2;
}

static std::vector<NativeMove> get_moves(const NativeBoard& b) {
    std::vector<NativeMove> moves;

    uint64_t workers_mask = bit_at(b.px, b.py) | bit_at(b.ox, b.oy);
    uint64_t blocked_cells = b.blocked_mask | b.primed_mask | workers_mask;
    bool can_prime = ((b.primed_mask | b.carpet_mask) & bit_at(b.px, b.py)) == 0;

    for (int d = 0; d < 4; ++d) {
        int nx = b.px + DX[d];
        int ny = b.py + DY[d];
        if (!on_board(nx, ny)) {
            continue;
        }

        uint64_t next_bit = bit_at(nx, ny);

        // Plain and prime can move onto carpet, but not blocked/primed/workers.
        if ((blocked_cells & next_bit) == 0) {
            moves.push_back({PLAIN_MT, d, 0});
            if (can_prime) {
                moves.push_back({PRIME_MT, d, 0});
            }
        }

        // Carpet moves must stay on a primed ray and cannot pass through workers.
        int cx = b.px;
        int cy = b.py;
        for (int roll = 1; roll < BOARD_N; ++roll) {
            cx += DX[d];
            cy += DY[d];
            if (!on_board(cx, cy)) {
                break;
            }
            if (cx == b.ox && cy == b.oy) {
                break;
            }
            if (!has_bit(b.primed_mask, cx, cy)) {
                break;
            }
            moves.push_back({CARPET_MT, d, roll});
        }
    }

    std::sort(moves.begin(), moves.end(), [](const NativeMove& a, const NativeMove& b) {
        int bucket_a = move_sort_bucket(a);
        int bucket_b = move_sort_bucket(b);
        if (bucket_a != bucket_b) {
            return bucket_a < bucket_b;
        }
        if (a.move_type == CARPET_MT && b.move_type == CARPET_MT && a.roll_length != b.roll_length) {
            return a.roll_length > b.roll_length;
        }
        if (a.direction != b.direction) {
            return a.direction < b.direction;
        }
        return a.roll_length > b.roll_length;
    });

    return moves;
}

static NativeBoard apply_and_flip(const NativeBoard& b, const NativeMove& m) {
    NativeBoard nb = b;

    if (m.move_type == PLAIN_MT) {
        nb.px += DX[m.direction];
        nb.py += DY[m.direction];
    } else if (m.move_type == PRIME_MT) {
        nb.primed_mask |= bit_at(nb.px, nb.py);
        nb.px += DX[m.direction];
        nb.py += DY[m.direction];
        nb.p_points += 1;
    } else if (m.move_type == CARPET_MT) {
        int cx = nb.px;
        int cy = nb.py;
        for (int i = 0; i < m.roll_length; ++i) {
            cx += DX[m.direction];
            cy += DY[m.direction];
            uint64_t bit = bit_at(cx, cy);
            nb.primed_mask &= ~bit;
            nb.carpet_mask |= bit;
        }
        nb.px = cx;
        nb.py = cy;
        nb.p_points += CPT[std::min(m.roll_length, 7)];
    }

    nb.p_turns -= 1;

    // Flip perspective so negamax always searches from the side to move.
    std::swap(nb.px, nb.ox);
    std::swap(nb.py, nb.oy);
    std::swap(nb.p_points, nb.o_points);
    std::swap(nb.p_turns, nb.o_turns);
    return nb;
}

static bool parse_belief_vector(PyObject* belief_obj, std::vector<double>& belief) {
    PyObject* seq = PySequence_Fast(belief_obj, "belief must be a sequence");
    if (seq == nullptr) {
        return false;
    }

    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    belief.resize(static_cast<size_t>(n));
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        belief[static_cast<size_t>(i)] = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
    }

    Py_DECREF(seq);
    return true;
}

static bool choose_search_candidate(
    const NativeBoard& b,
    const std::vector<double>& belief,
    NativeMove& search_move,
    float& search_score
) {
    if (belief.empty()) {
        return false;
    }

    auto it = std::max_element(belief.begin(), belief.end());
    double p = *it;
    if (p < SEARCH_PROB_THRESHOLD) {
        return false;
    }

    float ev = RAT_FIND_PTS * static_cast<float>(p) - RAT_MISS_PTS * (1.0f - static_cast<float>(p));
    int board_cost = best_carpet_run(b, b.px, b.py, b.ox, b.oy);
    if (board_cost > ev + 2.0f || ev <= 0.0f) {
        return false;
    }

    int idx = static_cast<int>(std::distance(belief.begin(), it));
    search_move = {SEARCH_MT, idx & 7, idx >> 3};
    search_score = ev;
    return true;
}

static float quiescence(
    const NativeBoard& b,
    float alpha,
    float beta,
    int qdepth,
    SearchContext& ctx
) {
    if (out_of_time(ctx)) {
        return 0.0f;
    }

    float stand_pat = eval_board(b);
    if (stand_pat >= beta) {
        return beta;
    }
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }
    if (qdepth <= 0) {
        return alpha;
    }

    auto moves = get_moves(b);
    for (const auto& m : moves) {
        // Extend only meaningful carpet tactics.
        if (m.move_type != CARPET_MT || m.roll_length < 2) {
            continue;
        }
        NativeBoard child = apply_and_flip(b, m);
        float score = -quiescence(child, -beta, -alpha, qdepth - 1, ctx);
        if (ctx.timed_out) {
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

static float negamax(
    const NativeBoard& b,
    int depth,
    float alpha,
    float beta,
    SearchContext& ctx
) {
    if (out_of_time(ctx)) {
        return 0.0f;
    }

    if (depth <= 0 || b.p_turns <= 0) {
        return quiescence(b, alpha, beta, QSEARCH_DEPTH, ctx);
    }

    auto moves = get_moves(b);
    if (moves.empty()) {
        return eval_board(b);
    }

    float best = -std::numeric_limits<float>::infinity();
    int n_searched = 0;

    for (const auto& m : moves) {
        if (out_of_time(ctx)) {
            return 0.0f;
        }

        NativeBoard child = apply_and_flip(b, m);
        float score = 0.0f;

        // Conservative LMR: only later plain moves get reduced first.
        bool use_lmr = (depth >= 3 && n_searched >= 2 && m.move_type == PLAIN_MT);
        if (use_lmr) {
            score = -negamax(child, depth - 2, -(alpha + 1.0f), -alpha, ctx);
            if (!ctx.timed_out && score > alpha) {
                score = -negamax(child, depth - 1, -beta, -alpha, ctx);
            }
        } else {
            score = -negamax(child, depth - 1, -beta, -alpha, ctx);
        }

        if (ctx.timed_out) {
            return 0.0f;
        }
        if (score > best) {
            best = score;
        }
        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) {
            break;
        }
        ++n_searched;
    }

    return best;
}

static bool search_root(
    const NativeBoard& root,
    const std::vector<double>& belief,
    int depth,
    SearchContext& ctx,
    RootResult& out
) {
    auto moves = get_moves(root);

    out.has_move = false;
    out.score = -INF;
    float alpha = -INF;
    float beta = INF;

    // Rat search is treated as one root-level candidate alongside board moves.
    NativeMove search_move{};
    float search_score = 0.0f;
    if (choose_search_candidate(root, belief, search_move, search_score)) {
        out.has_move = true;
        out.move = search_move;
        out.score = search_score;
        alpha = search_score;
    }

    if (moves.empty()) {
        return out.has_move;
    }

    for (const auto& m : moves) {
        if (out_of_time(ctx)) {
            return false;
        }
        NativeBoard child = apply_and_flip(root, m);
        float score = -negamax(child, depth - 1, -beta, -alpha, ctx);
        if (ctx.timed_out) {
            return false;
        }
        if (!out.has_move || score > out.score) {
            out.has_move = true;
            out.move = m;
            out.score = score;
        }
        if (score > alpha) {
            alpha = score;
        }
    }

    return out.has_move;
}

static PyObject* yolanda_search(PyObject* self, PyObject* args) {
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
    if (!parse_belief_vector(belief_obj, belief)) {
        return nullptr;
    }

    NativeBoard root{
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
    SearchContext ctx{};
    ctx.deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(safe_budget)
        );
    ctx.timed_out = false;

    RootResult best{false, {PLAIN_MT, 0, 0}, -INF};
    for (int depth = 1; depth <= MAX_DEPTH; ++depth) {
        RootResult current{};
        if (!search_root(root, belief, depth, ctx, current)) {
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
        yolanda_search,
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
