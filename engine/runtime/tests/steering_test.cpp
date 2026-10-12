#include "../steering_profile.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unistd.h>

using namespace g6rt::steering;
static int fails = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++fails; } } while (0)

static void put(const std::string& path, const std::string& bytes) {
    std::ofstream f(path, std::ios::binary); f.write(bytes.data(), std::streamsize(bytes.size()));
    if (!f) throw std::runtime_error("fixture write failed");
}

static std::string f32le(const std::vector<float>& x) {
    std::string out;
    for (float f : x) {
        const uint32_t b = float_bits(f);
        for (int k = 0; k < 4; ++k) out.push_back(char(b >> (8 * k)));
    }
    return out;
}

static std::string manifest(const std::string& tensor, std::string scope = "full", std::string layers = "[1,3]",
                            std::string alphas = "[1,3]", int bytes = 32) {
    const auto d = sha256(tensor.data(), tensor.size());
    const std::string z(64, '0'), o(64, '1'), t(64, '2'), f(64, 'f');
    return "{\n"
      "\"schema_version\":1,\"profile_name\":\"fixture\","
      "\"model\":{\"name\":\"glm-5.3\",\"revision\":\"rev\",\"weight_set\":\"weights\","
      "\"tokenizer_revision\":\"tok\",\"chat_template_sha256\":\"" + z + "\",\"hidden_size\":4,\"layer_count\":5},"
      "\"scope\":\"" + scope + "\",\"layers\":" + layers + ",\"alphas\":" + alphas + ","
      "\"directions\":{\"file\":\"directions.f32\",\"dtype\":\"float32-le\",\"byte_count\":" + std::to_string(bytes) +
      ",\"sha256\":\"" + g6::hex(d.data(), d.size()) + "\"},"
      "\"provenance\":{\"extraction_tool\":\"fixture\",\"extraction_commit\":\"abc\","
      "\"dataset_manifests\":{\"train\":\"" + o + "\",\"validation\":\"" + t + "\",\"final\":\"" + f + "\"}}}\n";
}

static ExpectedModel expected() { return {"glm-5.3", "rev", "weights", "tok", std::string(64, '0'), 4, 5}; }

static void expect_bad(const std::string& dir, const std::string& json, const std::string& tensor) {
    put(dir + "/manifest.json", json); put(dir + "/directions.f32", tensor);
    bool threw = false; try { (void)load_profile(dir, expected()); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

int main() {
    char tmp[] = "/tmp/g6-steering-test.XXXXXX";
    const char* made = mkdtemp(tmp); CHECK(made != nullptr); if (!made) return 1;
    const std::string dir = made;
    const std::string tensor = f32le({1,0,0,0, 0,1,0,0});
    put(dir + "/directions.f32", tensor); put(dir + "/manifest.json", manifest(tensor));
    auto p = load_profile(dir, expected());
    CHECK(p.schema_version == 1 && p.name == "fixture" && p.scope == Scope::Full);
    CHECK((p.layers == std::vector<uint32_t>{1,3}) && (p.alphas == std::vector<float>{1,3}));
    CHECK(p.directions.size() == 8 && p.directions[0] == 1 && p.directions[5] == 1);

    CHECK(canonical_scale_bits(-0.0) == 0 && canonical_scale_bits(0.0) == 0);
    CHECK(bits_float(canonical_scale_bits(1.25)) == 1.25f);
    bool threw = false; try { (void)canonical_scale_bits(NAN); } catch (...) { threw = true; } CHECK(threw);
    threw = false; try { (void)canonical_scale_bits(4.01); } catch (...) { threw = true; } CHECK(threw);
    const auto boot = BootConfig::from_profile(p);
    const auto off = boot.selected(canonical_scale_bits(-0.0));
    const auto on = boot.selected(canonical_scale_bits(1.0));
    CHECK(!off.enabled && on.enabled && off != on && on == on);
    uint64_t h0 = 14695981039346656037ull, h1 = h0; hash_identity(h0, off); hash_identity(h1, on); CHECK(h0 != h1);

    float aligned[] = {2,0,0,0}, d[] = {1,0,0,0}; project(aligned,d,4,1); CHECK(aligned[0] == 0);
    float orth[] = {0,2,3,4}; project(orth,d,4,3); CHECK(orth[0] == 0 && orth[1] == 2 && orth[3] == 4);
    float zero[] = {1,2,3,4}; const auto before = std::string(reinterpret_cast<char*>(zero), sizeof zero);
    project(zero,d,4,0); CHECK(before == std::string(reinterpret_cast<char*>(zero), sizeof zero));
    std::vector<float> shared{2,2,0,0}, full{5,2,0,0};
    project(shared.data(), d, shared.size(), 1); shared[0] += 3;
    project(full.data(), d, full.size(), 1);
    CHECK(shared[0] == 3 && shared[1] == 2); CHECK(full[0] == 0 && full[1] == 2);

    // Random rows use the same FP64 accumulation as the declared CPU oracle, including request_scale * layer_alpha.
    std::vector<float> rd(31), row(31), want(31); double norm2 = 0;
    uint32_t state = 0x6a09e667u;
    auto next = [&]() { state = state * 1664525u + 1013904223u; return int32_t(state) / double(INT32_MAX); };
    for (size_t i = 0; i < rd.size(); ++i) { rd[i] = float(next()); norm2 += double(rd[i]) * rd[i]; row[i] = float(next() * 1e4); }
    const double norm = std::sqrt(norm2); for (float& v : rd) v = float(v / norm);
    want = row; double dot = 0; for (size_t i = 0; i < row.size(); ++i) dot += double(row[i]) * rd[i];
    const float effective = 0.75f * 1.5f;
    for (size_t i = 0; i < want.size(); ++i) want[i] = float(double(want[i]) - double(effective) * dot * rd[i]);
    project(row.data(), rd.data(), row.size(), effective);
    for (size_t i = 0; i < row.size(); ++i) CHECK(std::fabs(row[i] - want[i]) <= 1e-6f * std::max(1.f, std::fabs(want[i])));

    // In real arithmetic projecting every TP partial equals projecting their sum. FP32 stores introduce bounded error.
    std::vector<float> pa(31), pb(31), parts(31), sum(31);
    for (size_t i = 0; i < pa.size(); ++i) { pa[i] = float(next() * 20); pb[i] = float(next() * 20); sum[i] = pa[i] + pb[i]; }
    project(pa.data(), rd.data(), pa.size(), .6f); project(pb.data(), rd.data(), pb.size(), .6f);
    for (size_t i = 0; i < parts.size(); ++i) parts[i] = pa[i] + pb[i];
    project(sum.data(), rd.data(), sum.size(), .6f);
    for (size_t i = 0; i < sum.size(); ++i) CHECK(std::fabs(parts[i] - sum[i]) <= 2e-5f * std::max(1.f, std::fabs(sum[i])));

    float extreme[] = {1e30f, -1e30f, 1e-30f, -1e-30f}, quarter[] = {.5f,.5f,.5f,.5f};
    project(extreme, quarter, 4, 4.f);
    for (float v : extreme) CHECK(std::isfinite(v));

    auto edit = [&](std::string from, std::string to) {
        auto s = manifest(tensor); const auto at = s.find(from); CHECK(at != std::string::npos);
        if (at != std::string::npos) s.replace(at, from.size(), to);
        return s;
    };
    expect_bad(dir, manifest(tensor, "other"), tensor);
    expect_bad(dir, manifest(tensor, "full", "[3,1]"), tensor);
    expect_bad(dir, manifest(tensor, "full", "[1,1]"), tensor);
    expect_bad(dir, manifest(tensor, "full", "[1,3]", "[1,5]"), tensor);
    expect_bad(dir, manifest(tensor, "full", "[1,3]", "[-1,3]"), tensor);
    expect_bad(dir, manifest(tensor, "full", "[1,3]", "[1]"), tensor);
    expect_bad(dir, manifest(tensor, "full", "[1,3]", "[1,3]", 28), tensor);
    expect_bad(dir, edit("\"dtype\":\"float32-le\"", "\"dtype\":\"float16-le\""), tensor);
    auto bad = tensor; bad[0] ^= 1; expect_bad(dir, manifest(tensor), bad);
    expect_bad(dir, manifest(tensor), tensor.substr(0, tensor.size() - 4));
    const auto bad_norm = f32le({2,0,0,0, 0,1,0,0}); expect_bad(dir, manifest(bad_norm), bad_norm);
    const auto zero_norm = f32le({0,0,0,0, 0,1,0,0}); expect_bad(dir, manifest(zero_norm), zero_norm);
    const auto nonfinite = f32le({NAN,0,0,0, 0,1,0,0}); expect_bad(dir, manifest(nonfinite), nonfinite);
    const auto infinite = f32le({INFINITY,0,0,0, 0,1,0,0}); expect_bad(dir, manifest(infinite), infinite);
    expect_bad(dir, edit("\"schema_version\":1", "\"schema_version\":2"), tensor);
    expect_bad(dir, edit("\"revision\":\"rev\"", "\"revision\":\"wrong\""), tensor);
    expect_bad(dir, edit("\"profile_name\":\"fixture\"", "\"profile_name\":\"bad name\""), tensor);
    expect_bad(dir, edit("\"scope\":\"full\",", ""), tensor);
    expect_bad(dir, edit("\"extraction_tool\":\"fixture\"", "\"extraction_tool\":\"\""), tensor);
    expect_bad(dir, edit("{\n\"schema_version\":1", "{\n\"schema_version\":1,\"schema_version\":1"), tensor);
    expect_bad(dir, edit("\"scope\":\"full\"", "\"scope\":\"full\",\"mystery\":1"), tensor);
    expect_bad(dir, manifest(tensor), tensor + "tail");

    std::filesystem::remove_all(dir);
    std::printf(fails ? "steering_test: %d FAIL\n" : "steering_test: PASS\n", fails);
    return fails != 0;
}
