#include "logger.h"

#include <ctime>
#include <fstream>
#include <iostream>
#include <utility>

// ============================================================================
// TRTLogger ʵ��
// ============================================================================

// [FIX] ����䣬noexcept ��ȫ
const char *TRTLogger::severityToString(Severity severity) noexcept {
  switch (severity) {
  case Severity::kINTERNAL_ERROR:
    return "INTERNAL_ERROR";
  case Severity::kERROR:
    return "ERROR";
  case Severity::kWARNING:
    return "WARNING";
  case Severity::kINFO:
    return "INFO";
  case Severity::kVERBOSE:
    return "VERBOSE";
  default:
    return "UNKNOWN";
  }
}

void TRTLogger::logWorker() {
  // ע�⣺mLogFilePath Ӧ�ڹ��캯���д��ݣ��˴�ͨ����Ա������ȡ
  // Ϊ���չʾ�������ѽ� logFilePath ��Ϊ��Ա���� mLogFilePath
  // ʵ��ʹ��ʱ������������ std::string mLogFilePath ��Ա
  std::ofstream logFile;

  if (!mLogFilePath.empty()) {
    logFile.open(mLogFilePath, std::ios::app);
    if (!logFile.is_open()) {
      std::cerr << "[TRT Logger] WARNING: Failed to open '" << mLogFilePath
                << "', logs will only appear on console.\n";
    }
  } else {
    std::cerr << "[TRT Logger] WARNING: Empty log file path, "
              << "logs will only appear on console.\n";
  }

  while (true) {
    std::unique_lock<std::mutex> lock(mQueueMutex);
    mCv.wait(lock, [this] { return !mLogQueue.empty() || mExit; });

    while (!mLogQueue.empty()) {
      std::string msg = std::move(mLogQueue.front());
      mLogQueue.pop();
      lock.unlock();

      std::cout << msg;

      if (logFile.is_open()) {
        logFile << msg;
        logFile.flush();
      }

      lock.lock();
    }

    if (mExit && mLogQueue.empty())
      break;
  }
}

TRTLogger::TRTLogger(Severity severity, const std::string &logFilePath)
    : mReportableSeverity(severity),
      mLogFilePath(logFilePath) // [FIX] �洢��־·��
      ,
      mExit(false) {
  mWorkerThread = std::thread(&TRTLogger::logWorker, this);
}

TRTLogger::~TRTLogger() {
  {
    std::lock_guard<std::mutex> lock(mQueueMutex);
    mExit = true;
  }
  mCv.notify_all(); // [FIX] notify_all �� notify_one ����ȫ
  if (mWorkerThread.joinable()) {
    mWorkerThread.join();
  }
}

void TRTLogger::log(Severity severity, const char *msg) noexcept {
  if (severity > mReportableSeverity.load(std::memory_order_relaxed))
    return;

  // [FIX] noexcept �����ڱ��벶�������쳣������ std::terminate
  try {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf{};

#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif

    char timestamp[64];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &tm_buf);

    // severityToString ���� const char*���˴���һ�� string ����
    std::string formattedMsg = "[" + std::string(timestamp) + "] [TRT] [" +
                               severityToString(severity) + "] " +
                               (msg ? msg : "(null)") + "\n"; // [FIX] ��ָ�����

    std::lock_guard<std::mutex> lock(mQueueMutex);
    mLogQueue.push(std::move(formattedMsg));
    mCv.notify_one();
  } catch (...) {
    // noexcept ��Լ�²����׳��κ��쳣
    // ��־ϵͳ������Ӧ��Ϊ����Դ����Ĭ�̵��쳣��Ψһ��ȷѡ��
  }
}

// ============================================================================
// TrLogger ʵ��
// ============================================================================

void TrLogger::log(Severity severity, const char *msg) noexcept {
  // [FIX] ͬ���汾ͬ����Ҫ noexcept ����
  try {
    const char *safeMsg = msg ? msg : "(null)";
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
    case Severity::kERROR:
      std::cerr << "[TRT ERROR] " << safeMsg << std::endl;
      break;
    case Severity::kWARNING:
      std::cerr << "[TRT WARN]  " << safeMsg << std::endl;
      break;
    case Severity::kINFO:
      std::cout << "[TRT INFO]  " << safeMsg << std::endl;
      break;
    default:
      break;
    }
  } catch (...) {
    // ��־���ʧ�ܲ�Ӧ���³�����ֹ
  }
}