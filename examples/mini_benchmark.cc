
//
// Created by joechen on 22-7-8.
// Update/Read workload
//

#include <time.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"

#include "utils.h"

using namespace rocksdb;

void CheckFreq(std::vector<uint64_t>& samples) {
  std::unordered_map<uint64_t, int> freqs;
  for (auto& value : samples) {
    ++freqs[value];
  }

  int cold_count = 0;
  for (auto& pair : freqs) {
    cold_count += (pair.second == 1);
  }

  int hot_count = freqs.size() - cold_count;
  float hot_freq =
      (samples.size() - cold_count) / (float)samples.size() * 100.f;
  printf("Update: hot count = %d(%.2f%%), cold count = %d(%.2f%%)\n", hot_count,
         hot_freq, cold_count, 100.f - hot_freq);
}

class KeyGenerator {
 public:
  KeyGenerator(int load_record_count, int run_record_count,
               uint64_t sample_range, double alpha = 0.98)
      : alpha_(alpha),
        interval_length_(sample_range / 100),
        sample_range_(sample_range),
        load_record_count_(load_record_count),
        run_record_count_(run_record_count),
        run_operation_count_(run_record_count) {
    std::random_device rd;
    std::random_device rd2;
    rng_ = std::default_random_engine{rd()};
    rng2_ = std::default_random_engine{rd2()};

    InitializeIntervals();
    Prepare();
  };

  void SetOperationCount(int run_operation_count) {
    if (run_operation_count > run_record_count_) {
      printf("Error: operation count must less than record count\n");
      abort();
    }
    run_operation_count_ = run_operation_count;
  }

  void SetReadRatio(double read_ratio) { read_ratio_ = read_ratio; }

  void SetPrefix(std::string& prefix) { prefix_ = prefix; }

  int GetLoadOperations() const { return load_record_count_; }

  int GetRunOperations() const { return run_operation_count_; }

  std::string NextLoadKey() {
    return GenerateKeyFromValue(load_records_[load_index_++]);
  }

  Operation Next(std::string& key) {
    if (rand_double(rng_) < read_ratio_) {
      key = NextReadKey();
      return kRead;
    } else {
      key = NextWriteKey();
      return kWrite;
    }
  }

 protected:
  void InitializeIntervals() {
    std::random_device shuffle_rd;
    std::vector<int> r(100);
    std::iota(r.begin(), r.end(), 0);
    std::shuffle(r.begin(), r.end(), shuffle_rd);

    memset(hot_read_intervals_, 0, sizeof(int) * 101);
    memset(hot_write_intervals_, 0, sizeof(int) * 101);
    for (int i = 0; i < 20; i++) {
      hot_read_intervals_[i] = 1;
      hot_write_intervals_[i + 80] = 1;
    }
    //    for (int i = 0; i < 10; ++i) {
    //      hot_read_intervals_[r[i]] = 1;
    //      hot_write_intervals_[r[i]] = 1;
    //    }
    //    for (int i = 10; i < 20; ++i) {
    //      hot_read_intervals_[r[i]] = 1;
    //    }
    //    for (int i = 20; i < 30; ++i) {
    //      hot_write_intervals_[r[i]] = 1;
    //    }

    printf("Hot write intervals:\n");
    for (int i = 0; i < 101; ++i) {
      if (hot_write_intervals_[i]) {
        printf("[%19lu, %19lu)\n", i * interval_length_,
               (i + 1) * interval_length_);
      }
    }

    printf("Hot write intervals:\n");
    for (int i = 0; i < 101; ++i) {
      if (hot_read_intervals_[i]) {
        printf("[%19lu, %19lu)\n", i * interval_length_,
               (i + 1) * interval_length_);
      }
    }
  }

  void Prepare() {
    if (!load_records_.empty()) {
      return;
    }

    // Generate load records
    load_index_.store(0, std::memory_order_relaxed);
    update_index_.store(0, std::memory_order_relaxed);
    load_records_.resize(load_record_count_);
    for (int i = 0; i < load_record_count_; ++i) {
      load_records_[i] = fnvhash64(i) % sample_range_;
    }

    // Generate zipfian data and map them to load records.
    // Extend load records if necessary.
    std::unordered_map<uint64_t, int> frequency = GenerateZipfianData();

    std::vector<uint64_t> hot_data;
    std::vector<uint64_t> cold_data;
    hot_data.reserve(run_records_.size());
    cold_data.reserve(run_records_.size());
    for (auto& pair : frequency) {
      (pair.second == 1 ? cold_data : hot_data).push_back(pair.first);
    }

    std::vector<uint64_t> hot_load_data;
    std::vector<uint64_t> cold_load_data;
    hot_load_data.reserve(run_records_.size());
    cold_load_data.reserve(run_records_.size());
    for (auto& v : load_records_) {
      (hot_write_intervals_[v / interval_length_] ? hot_load_data
                                                  : cold_load_data)
          .push_back(v);
    }

    std::random_device rd;
    std::default_random_engine rng = std::default_random_engine{rd()};
    std::uniform_int_distribution<uint64_t> dist(0, sample_range_ - 1);

    if (hot_load_data.size() < hot_data.size()) {
      int to_add = hot_data.size() - hot_load_data.size();
      load_records_.reserve(load_records_.size() + to_add);
      for (int i = 0; i < to_add; ++i) {
        uint64_t v = 0;
        do {
          v = dist(rng);
        } while (!hot_write_intervals_[v / interval_length_]);
        hot_load_data.push_back(v);
        load_records_.push_back(v);
      }
    }

    if (cold_load_data.size() < cold_data.size()) {
      int to_add = cold_data.size() - cold_load_data.size();
      load_records_.reserve(load_records_.size() + to_add);
      for (int i = 0; i < to_add; ++i) {
        uint64_t v = 0;
        do {
          v = dist(rng);
        } while (hot_write_intervals_[v / interval_length_]);
        cold_load_data.push_back(v);
        load_records_.push_back(v);
      }
    }

    std::random_device shuffle_rd;
    std::shuffle(hot_load_data.begin(), hot_load_data.end(), shuffle_rd);
    std::shuffle(cold_load_data.begin(), cold_load_data.end(), shuffle_rd);
    std::shuffle(load_records_.begin(), load_records_.end(), shuffle_rd);
    std::unordered_map<uint64_t, uint64_t> mapper;
    int index = 0;
    for (auto v : hot_data) {
      mapper[v] = hot_load_data[index++];
    }
    index = 0;
    for (auto v : cold_data) {
      mapper[v] = cold_load_data[index++];
    }

    for (auto& v : run_records_) {
      v = mapper[v];
    }

    hot_read_records_.reserve(load_records_.size());
    cold_read_records_.reserve(load_records_.size());
    for (auto v : load_records_) {
      (hot_read_intervals_[v / interval_length_] ? hot_read_records_
                                                 : cold_read_records_)
          .push_back(v);
    }

    printf("Change load_record_count from %d to %d\n", load_record_count_,
           (int)load_records_.size());
    load_record_count_ = load_records_.size();

    float hot_read_ratio =
        (float)(hot_read_records_.size()) * 100.f /
        (hot_read_records_.size() + cold_read_records_.size());
    printf("Read: hot count = %d(%.2f%%), cold count = %d(%.2f%%)\n",
           (int)hot_read_records_.size(), hot_read_ratio,
           (int)cold_read_records_.size(), 100.f - hot_read_ratio);

    CheckFreq(run_records_);
  }

  static uint64_t fnvhash64(int64_t val) {
    // from http://en.wikipedia.org/wiki/Fowler_Noll_Vo_hash
    static int64_t FNV_OFFSET_BASIS_64 = 0xCBF29CE484222325LL;
    static int64_t FNV_PRIME_64 = 1099511628211L;

    int64_t hashval = FNV_OFFSET_BASIS_64;

    for (int i = 0; i < 8; i++) {
      int64_t octet = val & 0x00ff;
      val = val >> 8;

      hashval = hashval ^ octet;
      hashval = hashval * FNV_PRIME_64;
      // hashval = hashval ^ octet;
    }
    return hashval > 0 ? hashval : -hashval;
  }

  std::unordered_map<uint64_t, int> GenerateZipfianData() {
    printf("Start generating zipfian data\n");
    double c = 0.0;

    uint64_t zipfian_sample_range = 1000000000;

    for (size_t i = 1; i <= zipfian_sample_range; i++) {
      c = c + (1.0 / pow((double)i, alpha_));
    }
    c = 1.0 / c;

    double* sum_probs = new double[zipfian_sample_range + 1];
    sum_probs[0] = 0;
    for (size_t i = 1; i <= zipfian_sample_range; ++i) {
      sum_probs[i] = sum_probs[i - 1] + c / pow((double)i, alpha_);
    }

    std::random_device rd;
    std::default_random_engine rng = std::default_random_engine{rd()};
    std::uniform_real_distribution<> rand_val{0.0, 1.0};

    std::unordered_map<uint64_t, int> frequency;
    run_records_.resize(run_record_count_);
    for (int i = 0; i < run_record_count_; ++i) {
      uint64_t res = 0;
      double z;
      do {
        z = rand_val(rng);
      } while ((z == 0) || (z == 1));

      uint64_t low = 1;
      uint64_t high = zipfian_sample_range;
      uint64_t mid;
      do {
        mid = (low + high) / 2;
        if (sum_probs[mid] >= z && sum_probs[mid - 1] < z) {
          res = mid;
          break;
        } else if (sum_probs[mid] >= z) {
          high = mid - 1;
        } else {
          low = mid + 1;
        }
      } while (low <= high);

      ++frequency[res];
      run_records_[i] = res;
    }

    return frequency;
  }

  std::string GenerateKeyFromValue(uint64_t value) {
    std::string s = std::to_string(value);
    return prefix_ + std::string(19 - s.length(), '0') + s;
  }

  std::string NextReadKey() {
    static std::uniform_int_distribution<uint64_t> rand1(
        0, hot_read_records_.size() - 1);
    static std::uniform_int_distribution<uint64_t> rand2(
        0, cold_read_records_.size() - 1);

    uint64_t v;
    if (rand_double(rng_) < 0.2) {
      v = hot_read_records_[rand1(rng2_)];
    } else {
      v = cold_read_records_[rand2(rng2_)];
    }

    return GenerateKeyFromValue(v);
  }

  std::string NextWriteKey() {
    return GenerateKeyFromValue(run_records_[update_index_++]);
  }

  double alpha_ = 0.98;

  double read_ratio_ = 0.5;

  int hot_read_intervals_[101];

  int hot_write_intervals_[101];

  uint64_t interval_length_;

  uint64_t sample_range_;

  int load_record_count_;

  int run_record_count_;

  int run_operation_count_;

  std::vector<uint64_t> load_records_;

  std::vector<uint64_t> run_records_;

  std::vector<uint64_t> hot_read_records_;

  std::vector<uint64_t> cold_read_records_;

  std::atomic<int> load_index_;

  std::atomic<int> update_index_;

  std::uniform_real_distribution<> rand_double{0.0, 1.0};

  std::default_random_engine rng_;

  std::default_random_engine rng2_;

  std::string prefix_ = "user";
};

class WorkloadRunner {
 public:
  WorkloadRunner(int num_threads, DB* db) : num_threads_(num_threads), db_(db) {
    work_threads_ = new std::thread[num_threads];
    thread_read_stats_ = new HistogramStat[num_threads];
    thread_write_stats_ = new HistogramStat[num_threads];
  }

  void SetKeyGenerator(KeyGenerator* generator) { key_generator_ = generator; }

  void Load() {
    completed_count_.store(0, std::memory_order_relaxed);

    for (int i = 0; i < num_threads_; ++i) {
      work_threads_[i] = std::thread(&WorkloadRunner::LoadPhase, this, i);
    }

    printf("Load opearation count = %d\n", key_generator_->GetLoadOperations());

    auto ops_thread = std::thread(&WorkloadRunner::StatisticsThread, this,
                                  key_generator_->GetLoadOperations());
    ops_thread.join();

    for (int i = 0; i < num_threads_; ++i) {
      work_threads_[i].join();
    }

    HistogramStat total_write_stat;
    for (int i = 0; i < num_threads_; ++i) {
      total_write_stat.Merge(thread_write_stats_[i]);
    }

    printf("Load done. Write Statistics: \n"
        "Avg(us): %.2lf\nMin(us): %zu\nMax(us): %zu\n"
        "90th(us): %.2lf\n99th(us): %.2lfn\n99.9th(us): %.2lf\n",
        total_write_stat.Median(), total_write_stat.max(), total_write_stat.max(),
        total_write_stat.Percentile(90), total_write_stat.Percentile(99),
        total_write_stat.Percentile(99.9));
  }

  void Run() {
    completed_count_.store(0, std::memory_order_relaxed);
    printf("Run opearation count = %d\n", key_generator_->GetRunOperations());

    for (int i = 0; i < num_threads_; ++i) {
      work_threads_[i] = std::thread(&WorkloadRunner::RunPhase, this, i);
    }

    auto ops_thread = std::thread(&WorkloadRunner::StatisticsThread, this,
                                  key_generator_->GetRunOperations());
    ops_thread.join();

    for (int i = 0; i < num_threads_; ++i) {
      work_threads_[i].join();
    }

    printf("Run phase done, failed count = %d\n",
           failed_count_.load(std::memory_order_relaxed));

    HistogramStat total_write_stat;
    HistogramStat total_read_stat;
    for (int i = 0; i < num_threads_; ++i) {
      total_write_stat.Merge(thread_write_stats_[i]);
      total_read_stat.Merge(thread_read_stats_[i]);
    }

    printf("Write Statistics: \n"
        "Avg(us): %.1lf\nMin(us): %zu\nMax(us): %zu\n"
        "90th(us): %.1lf\n99th(us): %.1lfn\n99.9th(us): %.1lf\n",
        total_write_stat.Median(), total_write_stat.max(), total_write_stat.max(),
        total_write_stat.Percentile(90), total_write_stat.Percentile(99),
        total_write_stat.Percentile(99.9));
    printf("Read Statistics: \n"
        "Avg(us): %.1lf\nMin(us): %zu\nMax(us): %zu\n"
        "90th(us): %.1lf\n99th(us): %.1lfn\n99.9th(us): %.1lf\n",
        total_read_stat.Median(), total_read_stat.max(), total_read_stat.max(),
        total_read_stat.Percentile(90), total_read_stat.Percentile(99),
        total_read_stat.Percentile(99.9));
  }

  void SetMetricInterval(int interval) { interval_ = interval; }

  static std::string GenerateValueFromKey(std::string& key) {
    int repeat_times = 1024UL / key.length();
    size_t len = key.length() * repeat_times;
    std::string value;
    value.reserve(1024);
    while (repeat_times--) {
      value += key;
    }
    value += key.substr(0, 1024 - value.length());
    return value;
  }

 private:
  // TODO: need latency statistics
  void LoadPhase(int thread_id) {
    int operation_count = key_generator_->GetLoadOperations();
    int operation_per_thread = operation_count / num_threads_;
    if (thread_id == 0) {
      operation_per_thread += (operation_count % num_threads_);
    }

    Status s;
    decltype(std::chrono::steady_clock::now()) start_time, end_time;
    uint64_t latency = 0;

    while (operation_per_thread--) {
      auto key = key_generator_->NextLoadKey();
      auto value = GenerateValueFromKey(key);
      start_time = std::chrono::steady_clock::now();
      s = db_->Put(WriteOptions(), key, value);
      end_time = std::chrono::steady_clock::now();
      latency = std::chrono::duration_cast<std::chrono::microseconds>
                (end_time - start_time).count();
      thread_write_stats_[thread_id].Add(latency);
      ++completed_count_;
      assert(s.ok() || s.IsNotFound());
    }
  }

  void RunPhase(int thread_id) {
    int operation_count = key_generator_->GetRunOperations();
    int operation_per_thread = operation_count / num_threads_;
    if (thread_id == 0) {
      operation_per_thread += (operation_count % num_threads_);
    }

    std::string key;
    std::string ret;
    Status s;

    decltype(std::chrono::steady_clock::now()) start_time, end_time;
    uint64_t latency = 0;

    while (operation_per_thread--) {
      auto type = key_generator_->Next(key);
      auto value = GenerateValueFromKey(key);

      switch (type) {
        case kWrite:
          start_time = std::chrono::steady_clock::now();
          s = db_->Put(WriteOptions(), key, value);
          end_time = std::chrono::steady_clock::now();
          latency = std::chrono::duration_cast<std::chrono::microseconds>
                    (end_time - start_time).count();
          thread_write_stats_[thread_id].Add(latency);
          break;
        case kRead:
          start_time = std::chrono::steady_clock::now();
          s = db_->Get(ReadOptions(), key, &ret);
          end_time = std::chrono::steady_clock::now();
          latency = std::chrono::duration_cast<std::chrono::microseconds>
                    (end_time - start_time).count();
          thread_read_stats_[thread_id].Add(latency);
          break;
      }

      ++completed_count_;
      if (s.IsNotFound()) {
        ++failed_count_;
      }
      assert(s.ok() || s.IsNotFound());
    }
  }

  void StatisticsThread(int operation_counts) {
    int prev_completed = 0;
    int seconds = 0;
    while (prev_completed < operation_counts) {
      int new_completed = completed_count_.load(std::memory_order_relaxed);
      int ops = (new_completed - prev_completed) / interval_;
      printf("[%d sec] %d operations; %d current ops/sec\n", seconds,
             new_completed, ops);

      prev_completed = new_completed;
      std::this_thread::sleep_for(std::chrono::seconds(interval_));
      seconds += interval_;
    }
  }

  int interval_ = 5;

  int num_threads_ = 16;

  std::thread* work_threads_;

  std::thread ops_thread_;

  std::atomic<int> completed_count_{0};

  std::atomic<int> failed_count_{0};

  KeyGenerator* key_generator_ = nullptr;

  DB* db_;

  HistogramStat* thread_write_stats_;
  HistogramStat* thread_read_stats_;
};

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);

  float read_ratio = 0.5f;
  if (argc == 2) {
    read_ratio = atof(argv[1]);
  }

  printf("Read ratio = %.2f\n", read_ratio);

  int thread_num = 8;
  int load_count = 80000000;
  int run_count = 320000000;
  uint64_t sample_range = 9000000000000000000;

  Options options;
  options.create_if_missing = true;
  options.use_direct_io_for_flush_and_compaction = true;
  options.use_direct_reads = true;
  options.enable_pipelined_write = true;
  options.compression = rocksdb::kNoCompression;
  options.nvm_path = "/pg_wal/ycc/memory_art";
  options.IncreaseParallelism(16);

  std::remove(options.nvm_path.c_str());

  std::string db_path = "/tmp/tmp_data/db_art";

  DB* db;
  DB::Open(options, db_path, &db);

  auto gen = new KeyGenerator(load_count, run_count, sample_range, 0.98);
  gen->SetReadRatio(0.00000000001);

  WorkloadRunner runner(thread_num, db);
  runner.SetKeyGenerator(gen);
  runner.SetMetricInterval(2);
  //runner.Load();
  runner.Run();

  db->Close();

  delete gen;
  delete db;
}