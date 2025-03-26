#include <stddef.h>
#include <stdint.h>
#include <cmath>
#include <iostream>
#include <cassert>

class BloomMath {
 public:
  // False positive rate of a standard Bloom filter, for given ratio of
  // filter memory bits to added keys, and number of probes per operation.
  // (The false positive rate is effectively independent of scale, assuming
  // the implementation scales OK.)
  static double StandardFpRate(double bits_per_key, int num_probes) {
    // Standard very-good-estimate formula. See
    // https://en.wikipedia.org/wiki/Bloom_filter#Probability_of_false_positives
    return std::pow(1.0 - std::exp(-num_probes / bits_per_key), num_probes);
  }

  // False positive rate of a "blocked"/"shareded"/"cache-local" Bloom filter,
  // for given ratio of filter memory bits to added keys, number of probes per
  // operation (all within the given block or cache line size), and block or
  // cache line size.
  static double CacheLocalFpRate(double bits_per_key, int num_probes,
                                 int cache_line_bits) {
    double keys_per_cache_line = cache_line_bits / bits_per_key;
    // A reasonable estimate is the average of the FP rates for one standard
    // deviation above and below the mean bucket occupancy. See
    // https://github.com/facebook/rocksdb/wiki/RocksDB-Bloom-Filter#the-math
    double keys_stddev = std::sqrt(keys_per_cache_line);
    double crowded_fp = StandardFpRate(
        cache_line_bits / (keys_per_cache_line + keys_stddev), num_probes);
    double uncrowded_fp = StandardFpRate(
        cache_line_bits / (keys_per_cache_line - keys_stddev), num_probes);
    return (crowded_fp + uncrowded_fp) / 2;
  }

  // False positive rate of querying a new item against `num_keys` items, all
  // hashed to `fingerprint_bits` bits. (This assumes the fingerprint hashes
  // themselves are stored losslessly. See Section 4 of
  // http://www.ccs.neu.edu/home/pete/pub/bloom-filters-verification.pdf)
  static double FingerprintFpRate(size_t num_keys, int fingerprint_bits) {
    double inv_fingerprint_space = std::pow(0.5, fingerprint_bits);
    // Base estimate assumes each key maps to a unique fingerprint.
    // Could be > 1 in extreme cases.
    double base_estimate = num_keys * inv_fingerprint_space;
    // To account for potential overlap, we choose between two formulas
    if (base_estimate > 0.0001) {
      // A very good formula assuming we don't construct a floating point
      // number extremely close to 1. Always produces a probability < 1.
      return 1.0 - std::exp(-base_estimate);
    } else {
      // A very good formula when base_estimate is far below 1. (Subtract
      // away the integral-approximated sum that some key has same hash as
      // one coming before it in a list.)
      return base_estimate - (base_estimate * base_estimate * 0.5);
    }
  }

  // Returns the probably of either of two independent(-ish) events
  // happening, given their probabilities. (This is useful for combining
  // results from StandardFpRate or CacheLocalFpRate with FingerprintFpRate
  // for a hash-efficient Bloom filter's FP rate. See Section 4 of
  // http://www.ccs.neu.edu/home/pete/pub/bloom-filters-verification.pdf)
  static double IndependentProbabilitySum(double rate1, double rate2) {
    // Use formula that avoids floating point extremely close to 1 if
    // rates are extremely small.
    return rate1 + rate2 - (rate1 * rate2);
  }
};

template <bool ExtraRotates>
class LegacyBloomImpl {
 public:
  static void CompareFpRate(size_t keys, int bits_per_key) {
    size_t bytes = keys * bits_per_key / 8;
    EstimatedFpRate(keys, bytes, ChooseNumProbes(bits_per_key));
  }

  // NOTE: this has only been validated to enough accuracy for producing
  // reasonable warnings / user feedback, not for making functional decisions.
  static void EstimatedFpRate(size_t keys, size_t bytes, int num_probes) {
    double bits_per_key = 8.0 * bytes / keys;
    double filter_rate = BloomMath::CacheLocalFpRate(bits_per_key, num_probes,
                                                     /*cache line bits*/ 512);
    if (!ExtraRotates) {
      // Good estimate of impact of flaw in index computation.
      // Adds roughly 0.002 around 50 bits/key and 0.001 around 100 bits/key.
      // The + 22 shifts it nicely to fit for lower bits/key.
      filter_rate += 0.1 / (bits_per_key * 0.75 + 22);
    } else {
      // Not yet validated
      assert(false);
    }
    // Always uses 32-bit hash
    double fingerprint_rate = BloomMath::FingerprintFpRate(keys, 32);
    double combined_rate = BloomMath::IndependentProbabilitySum(filter_rate, fingerprint_rate);
    double standard_rate = BloomMath::StandardFpRate(bits_per_key, num_probes);

    std::cout << "keys : " << keys << ", bytes : " << bytes << ", bits per key : " << bits_per_key << std::endl;
    std::cout << "standard FPR : " << standard_rate << std::endl;
    std::cout << "cache local FPR : " << filter_rate << std::endl;
    std::cout << "fingerprint FPR : " << fingerprint_rate << std::endl;
    std::cout << "combined FPR : " << combined_rate << std::endl;
    std::cout << std::endl << std::endl;
  }

  static inline int ChooseNumProbes(int bits_per_key) {
    // We intentionally round down to reduce probing cost a little bit
    int num_probes = static_cast<int>(bits_per_key * 0.69);  // 0.69 =~ ln(2)
    if (num_probes < 1) num_probes = 1;
    if (num_probes > 30) num_probes = 30;
    return num_probes;
  }
};

int main() {
    LegacyBloomImpl<false> Bloom;
    Bloom.CompareFpRate(1024 * 1024, 16);
}