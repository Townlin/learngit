// Bit-sliced simulator for kickmix .kmx circuits with per-gate instrumentation.
//
// Semantics follow the published kickmix gate set (classical reversible gates, phase gates,
// HMR = X-basis measure+reset, R = reset with dephasing, classically conditioned variants).
// Besides checking the DQI contract (syndrome restored, ancillas zero, phase), it records for
// every Toffoli-like gate (CCX / CCZ / CSWAP) how often it is executed and how often it
// actually changes the state ("fires"), and can run with a set of gates removed.
//
// usage: kmxsim circuit.kmx inputs.bin out_prefix [--skip skiplist.txt] [--threads N]
// inputs.bin: int64 num_shots (multiple of 64), then bit-packed (little endian, along shots)
//             rows: 840 syndrome rows, 49140 mask rows, 1 expected-phase row.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

enum Op : uint8_t { CX, CCX, CZ, CCZ, SWAP, CSWAP, X, Z, NEG, HMR, RST, BINV, BST0, BST1, PUSH, POP, TAG };

struct Ins {
    uint8_t op;
    uint8_t has_cond;
    uint16_t q0, q1, q2;
    uint32_t b;     // target bit (HMR, BIT_*) or tag id (TAG)
    uint32_t cond;  // condition bit
};

static inline uint32_t parse_id(const char *&p) {  // parses "q123" or "b123" or "r12"
    while (*p == ' ') p++;
    p++;  // skip letter
    uint32_t v = 0;
    while (*p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
    return v;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: kmxsim circuit.kmx inputs.bin out_prefix [--skip file] [--threads N]\n");
        return 2;
    }
    std::string skip_path;
    int threads = 4;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--skip") && i + 1 < argc) skip_path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    }
    std::unordered_set<uint64_t> skip;
    if (!skip_path.empty()) {
        std::ifstream f(skip_path);
        uint64_t v;
        while (f >> v) skip.insert(v);
    }

    // ---- parse circuit ----
    std::vector<Ins> prog;
    prog.reserve(60000000);
    std::vector<uint32_t> toff_line;                 // toffoli index -> op index in prog
    std::vector<std::string> tag_names;
    std::map<std::string, uint32_t> tag_ids;
    std::vector<std::vector<uint32_t>> reg_members;  // register id -> ids (qubit or bit)
    std::vector<char> reg_kind;                      // 'q' or 'b'
    std::map<std::string, uint32_t> reg_by_name;
    uint32_t num_q = 0, num_b = 0;
    {
        FILE *f = fopen(argv[1], "r");
        if (!f) { perror("circuit"); return 2; }
        char *line = nullptr;
        size_t cap = 0;
        ssize_t len;
        uint64_t toff_seen = 0;
        while ((len = getline(&line, &cap, f)) > 0) {
            const char *p = line;
            Ins in{};
            const char *sp = strchr(p, ' ');
            size_t n = sp ? (size_t)(sp - p) : strcspn(p, "\n");
            auto is = [&](const char *s) { return strlen(s) == n && !strncmp(p, s, n); };
            const char *a = p + n;
            const char *ifp = strstr(a, " if ");
            if (ifp) { in.has_cond = 1; const char *c = ifp + 4; in.cond = parse_id(c); num_b = std::max(num_b, in.cond + 1); }
            bool toff = false;
            if (is("CX")) { in.op = CX; in.q0 = parse_id(a); in.q1 = parse_id(a); }
            else if (is("CCX")) { in.op = CCX; in.q0 = parse_id(a); in.q1 = parse_id(a); in.q2 = parse_id(a); toff = true; }
            else if (is("CZ")) { in.op = CZ; in.q0 = parse_id(a); in.q1 = parse_id(a); }
            else if (is("CCZ")) { in.op = CCZ; in.q0 = parse_id(a); in.q1 = parse_id(a); in.q2 = parse_id(a); toff = true; }
            else if (is("SWAP")) { in.op = SWAP; in.q0 = parse_id(a); in.q1 = parse_id(a); }
            else if (is("CSWAP")) { in.op = CSWAP; in.q0 = parse_id(a); in.q1 = parse_id(a); in.q2 = parse_id(a); toff = true; }
            else if (is("X")) { in.op = X; in.q0 = parse_id(a); }
            else if (is("Z")) { in.op = Z; in.q0 = parse_id(a); }
            else if (is("NEG")) { in.op = NEG; }
            else if (is("HMR")) { in.op = HMR; in.q0 = parse_id(a); in.b = parse_id(a); num_b = std::max(num_b, in.b + 1); }
            else if (is("R")) { in.op = RST; in.q0 = parse_id(a); }
            else if (is("BIT_INVERT")) { in.op = BINV; in.b = parse_id(a); num_b = std::max(num_b, in.b + 1); }
            else if (is("BIT_STORE0")) { in.op = BST0; in.b = parse_id(a); num_b = std::max(num_b, in.b + 1); }
            else if (is("BIT_STORE1")) { in.op = BST1; in.b = parse_id(a); num_b = std::max(num_b, in.b + 1); }
            else if (is("PUSH_CONDITION")) { in.op = PUSH; }
            else if (is("POP_CONDITION")) { in.op = POP; }
            else if (is("INFO_TAG")) {
                in.op = TAG;
                const char *q1 = strchr(a, '"');
                const char *q2 = q1 ? strrchr(a, '"') : nullptr;
                std::string s = (q1 && q2 > q1) ? std::string(q1 + 1, q2) : std::string();
                auto it = tag_ids.find(s);
                if (it == tag_ids.end()) { it = tag_ids.emplace(s, (uint32_t)tag_names.size()).first; tag_names.push_back(s); }
                in.b = it->second;
            } else if (is("APPEND_TO_REGISTER")) {
                while (*a == ' ') a++;
                char kind = *a;
                uint32_t id = parse_id(a);
                uint32_t r = parse_id(a);
                if (reg_members.size() <= r) { reg_members.resize(r + 1); reg_kind.resize(r + 1, '?'); }
                reg_members[r].push_back(id);
                reg_kind[r] = kind;
                if (kind == 'q') num_q = std::max(num_q, id + 1); else num_b = std::max(num_b, id + 1);
                continue;
            } else if (is("REGISTER")) {
                uint32_t r = parse_id(a);
                const char *q1 = strchr(a, '"');
                const char *q2 = strrchr(a, '"');
                reg_by_name[std::string(q1 + 1, q2)] = r;
                continue;
            } else {
                fprintf(stderr, "unknown op: %s", line);
                return 2;
            }
            num_q = std::max<uint32_t>(num_q, std::max({in.q0, in.q1, in.q2}) + 1);
            if (toff) {
                if (skip.count(toff_seen)) { toff_seen++; continue; }
                toff_line.push_back((uint32_t)prog.size());
                toff_seen++;
            }
            prog.push_back(in);
        }
        free(line);
        fclose(f);
    }
    fprintf(stderr, "parsed %zu ops, %zu toffoli-like kept, %u qubits, %u bits\n", prog.size(), toff_line.size(), num_q, num_b);

    std::vector<uint32_t> sq, eb;
    for (int i = 0; i < 70; i++) { auto &m = reg_members[reg_by_name.at("S" + std::to_string(i))]; sq.insert(sq.end(), m.begin(), m.end()); }
    for (int j = 0; j < 4095; j++) { auto &m = reg_members[reg_by_name.at("E" + std::to_string(j))]; eb.insert(eb.end(), m.begin(), m.end()); }
    std::vector<char> is_s(num_q, 0);
    for (auto q : sq) is_s[q] = 1;

    // ---- inputs ----
    FILE *fi = fopen(argv[2], "rb");
    if (!fi) { perror("inputs"); return 2; }
    int64_t shots;
    if (fread(&shots, 8, 1, fi) != 1) return 2;
    size_t W = shots / 64;
    std::vector<uint64_t> in_s(840 * W), in_e(49140 * W), in_p(W);
    if (fread(in_s.data(), 8, in_s.size(), fi) != in_s.size()) return 2;
    if (fread(in_e.data(), 8, in_e.size(), fi) != in_e.size()) return 2;
    if (fread(in_p.data(), 8, in_p.size(), fi) != in_p.size()) return 2;
    fclose(fi);

    // toffoli index lookup: op index -> toffoli slot
    std::vector<uint32_t> exec_cnt(toff_line.size(), 0), fire_cnt(toff_line.size(), 0);
    std::vector<std::atomic<uint64_t>> dummy(0);
    std::vector<uint64_t> fail_val(W, 0), fail_phase(W, 0);
    std::atomic<uint64_t> reset_dirty{0};

    // process in chunks of CW words per thread
    const size_t CW = 8;  // 512 shots per chunk
    std::atomic<size_t> next{0};
    std::vector<std::vector<uint32_t>> t_exec(threads), t_fire(threads);
    auto worker = [&](int tid) {
        auto &ex = t_exec[tid];
        auto &fr = t_fire[tid];
        ex.assign(toff_line.size(), 0);
        fr.assign(toff_line.size(), 0);
        std::mt19937_64 rng(0x9E3779B97F4A7C15ULL * (tid + 1));
        std::vector<uint64_t> qs((size_t)num_q * CW), bs((size_t)num_b * CW);
        std::vector<uint64_t> cstack;
        while (true) {
            size_t w0 = next.fetch_add(CW);
            if (w0 >= W) break;
            size_t cw = std::min(CW, W - w0);
            std::fill(qs.begin(), qs.end(), 0);
            std::fill(bs.begin(), bs.end(), 0);
            for (size_t i = 0; i < sq.size(); i++) for (size_t k = 0; k < cw; k++) qs[sq[i] * CW + k] = in_s[i * W + w0 + k];
            for (size_t i = 0; i < eb.size(); i++) for (size_t k = 0; k < cw; k++) bs[eb[i] * CW + k] = in_e[i * W + w0 + k];
            uint64_t phase[CW] = {0}, cond[CW];
            for (size_t k = 0; k < CW; k++) cond[k] = ~0ULL;
            cstack.clear();
            size_t ti = 0;
            uint64_t dirty = 0;
            for (size_t pc = 0; pc < prog.size(); pc++) {
                const Ins &in = prog[pc];
                uint64_t c[CW];
                if (in.has_cond) { const uint64_t *cb = &bs[(size_t)in.cond * CW]; for (size_t k = 0; k < CW; k++) c[k] = cond[k] & cb[k]; }
                else { for (size_t k = 0; k < CW; k++) c[k] = cond[k]; }
                uint64_t *a = &qs[(size_t)in.q0 * CW], *b2 = &qs[(size_t)in.q1 * CW], *d = &qs[(size_t)in.q2 * CW];
                switch (in.op) {
                    case CX: for (size_t k = 0; k < CW; k++) b2[k] ^= c[k] & a[k]; break;
                    case CCX: {
                        uint32_t e = 0, f = 0;
                        for (size_t k = 0; k < CW; k++) { uint64_t t = c[k] & a[k] & b2[k]; d[k] ^= t; e += __builtin_popcountll(c[k]); f += __builtin_popcountll(t); }
                        ex[ti] += e; fr[ti] += f; ti++;
                        break;
                    }
                    case CZ: for (size_t k = 0; k < CW; k++) phase[k] ^= c[k] & a[k] & b2[k]; break;
                    case CCZ: {
                        uint32_t e = 0, f = 0;
                        for (size_t k = 0; k < CW; k++) { uint64_t t = c[k] & a[k] & b2[k] & d[k]; phase[k] ^= t; e += __builtin_popcountll(c[k]); f += __builtin_popcountll(t); }
                        ex[ti] += e; fr[ti] += f; ti++;
                        break;
                    }
                    case SWAP: for (size_t k = 0; k < CW; k++) { uint64_t t = (a[k] ^ b2[k]) & c[k]; a[k] ^= t; b2[k] ^= t; } break;
                    case CSWAP: {
                        uint32_t e = 0, f = 0;
                        for (size_t k = 0; k < CW; k++) { uint64_t t = (b2[k] ^ d[k]) & a[k] & c[k]; b2[k] ^= t; d[k] ^= t; e += __builtin_popcountll(c[k]); f += __builtin_popcountll(t); }
                        ex[ti] += e; fr[ti] += f; ti++;
                        break;
                    }
                    case X: for (size_t k = 0; k < CW; k++) a[k] ^= c[k]; break;
                    case Z: for (size_t k = 0; k < CW; k++) phase[k] ^= c[k] & a[k]; break;
                    case NEG: for (size_t k = 0; k < CW; k++) phase[k] ^= c[k]; break;
                    case HMR: {
                        uint64_t *bt = &bs[(size_t)in.b * CW];
                        for (size_t k = 0; k < CW; k++) { uint64_t r = rng(); uint64_t nb = (bt[k] & ~c[k]) | (r & c[k]); phase[k] ^= a[k] & nb & c[k]; a[k] &= ~c[k]; bt[k] = nb; }
                        break;
                    }
                    case RST:
                        for (size_t k = 0; k < CW; k++) { uint64_t r = rng(); dirty += __builtin_popcountll(a[k] & c[k]); phase[k] ^= a[k] & r & c[k]; a[k] &= ~c[k]; }
                        break;
                    case BINV: { uint64_t *bt = &bs[(size_t)in.b * CW]; for (size_t k = 0; k < CW; k++) bt[k] ^= c[k]; break; }
                    case BST0: { uint64_t *bt = &bs[(size_t)in.b * CW]; for (size_t k = 0; k < CW; k++) bt[k] &= ~c[k]; break; }
                    case BST1: { uint64_t *bt = &bs[(size_t)in.b * CW]; for (size_t k = 0; k < CW; k++) bt[k] |= c[k]; break; }
                    case PUSH:
                        for (size_t k = 0; k < CW; k++) { cstack.push_back(cond[k]); cond[k] = c[k]; }
                        break;
                    case POP:
                        for (size_t k = CW; k-- > 0;) { cond[k] = cstack.back(); cstack.pop_back(); }
                        break;
                    case TAG: break;
                }
            }
            reset_dirty += dirty;
            // checks
            for (size_t k = 0; k < cw; k++) {
                uint64_t ve = 0;
                for (size_t i = 0; i < sq.size(); i++) ve |= qs[sq[i] * CW + k] ^ in_s[i * W + w0 + k];
                for (uint32_t q = 0; q < num_q; q++) if (!is_s[q]) ve |= qs[(size_t)q * CW + k];
                fail_val[w0 + k] = ve;
                fail_phase[w0 + k] = (phase[k] ^ in_p[w0 + k]) & ~ve;
            }
        }
    };
    std::vector<std::thread> th;
    for (int t = 0; t < threads; t++) th.emplace_back(worker, t);
    for (auto &t : th) t.join();
    for (int t = 0; t < threads; t++)
        for (size_t i = 0; i < toff_line.size(); i++) { exec_cnt[i] += t_exec[t][i]; fire_cnt[i] += t_fire[t][i]; }

    uint64_t nv = 0, np = 0, tot_exec = 0;
    for (size_t k = 0; k < W; k++) { nv += __builtin_popcountll(fail_val[k]); np += __builtin_popcountll(fail_phase[k]); }
    for (auto e : exec_cnt) tot_exec += e;
    printf("{\"shots\": %lld, \"value_errors\": %llu, \"phase_errors\": %llu, \"toffoli_mean\": %.6f, \"toffoli_static\": %zu, \"reset_dirty\": %llu, \"qubits\": %u}\n",
           (long long)shots, (unsigned long long)nv, (unsigned long long)np, (double)tot_exec / shots, toff_line.size(),
           (unsigned long long)reset_dirty.load(), num_q);

    // per-gate stats with tag path
    std::string outp = std::string(argv[3]) + ".gates.tsv";
    FILE *fo = fopen(outp.c_str(), "w");
    std::vector<uint32_t> stack;
    size_t ti = 0;
    uint64_t orig_idx = 0;  // index among original toffolis (accounting for skipped ones is done by caller)
    for (size_t pc = 0; pc < prog.size(); pc++) {
        const Ins &in = prog[pc];
        if (in.op == TAG) {
            const std::string &s = tag_names[in.b];
            if (s.rfind("@enter:", 0) == 0) stack.push_back(in.b);
            else if (s.rfind("@exit:", 0) == 0 && !stack.empty()) stack.pop_back();
        } else if (in.op == CCX || in.op == CCZ || in.op == CSWAP) {
            std::string path;
            for (auto t : stack) { path += tag_names[t].substr(7); path += '/'; }
            fprintf(fo, "%zu\t%zu\t%s\t%u\t%u\t%u\t%u\t%u\t%s\n", ti, pc, in.op == CCX ? "CCX" : in.op == CCZ ? "CCZ" : "CSWAP",
                    in.q0, in.q1, in.q2, exec_cnt[ti], fire_cnt[ti], path.c_str());
            ti++;
        }
    }
    fclose(fo);
    // failing shots
    std::string outf = std::string(argv[3]) + ".fails.txt";
    FILE *ff = fopen(outf.c_str(), "w");
    for (size_t k = 0; k < W; k++)
        for (int bit = 0; bit < 64; bit++) {
            bool v = (fail_val[k] >> bit) & 1, p = (fail_phase[k] >> bit) & 1;
            if (v || p) fprintf(ff, "%zu\t%c\n", k * 64 + bit, v ? 'V' : 'P');
        }
    fclose(ff);
    (void)orig_idx;
    return 0;
}
