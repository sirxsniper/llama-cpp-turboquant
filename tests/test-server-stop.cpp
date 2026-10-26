// [TAG_FN_V0254_STOP] a server stop while a request waits for its result (tools/server/server-queue.cpp): after
// server_response::stop_waiters() every waiting reader returns an error result at once (a non-streaming wait_for_all, a
// streaming next, a blocking recv) and a reader that starts later gets one too, so no HTTP worker waits for a task the
// ended loop never finishes. Before the stop a real result arrives as usual. CPU only, no model, no task loop.
#include "server-queue.h"
#include "server-task.h"

#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <thread>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                      \
    do {                                       \
        g_checks++;                            \
        if (!(cond)) {                         \
            g_fail++;                          \
            fprintf(stderr, "FAIL: ");         \
            fprintf(stderr, __VA_ARGS__);      \
            fprintf(stderr, "\n");             \
        }                                      \
    } while (0)

using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// a reader that never returns would hang the test: give up loudly instead
template <typename T>
static T wait_or_die(std::future<T> & f, const char * what) {
    if (f.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
        fprintf(stderr, "FAIL: %s still waiting 10 s after the stop (the HTTP worker would hang)\n", what);
        fflush(stderr);
        std::_Exit(1);
    }
    return f.get();
}

static bool is_stop_error(const server_task_result_ptr & r) {
    if (r == nullptr || !r->is_error()) {
        return false;
    }
    const json j = r->to_json();
    return j.value("code", 0) == 503 && j.value("type", std::string()) == "unavailable_error";
}

static void post_metrics(server_response_reader & rd) {
    server_task task(SERVER_TASK_TYPE_METRICS);
    task.id = rd.get_new_id();
    rd.post_task(std::move(task));
}

int main() {
    ggml_time_init(); // the queues stamp tasks with ggml_time_ms (llama_backend_init does this in the server)

    server_queue    queue_tasks; // no loop runs: a posted task is never processed, as after a stop
    server_response queue_results;
    const auto connected = [] { return false; }; // the client keeps its connection open

    // 1. before the stop: a real result arrives, nothing else
    {
        server_response_reader rd(queue_tasks, queue_results, 1);
        post_metrics(rd);
        const int id = *rd.id_tasks.begin();
        std::thread sender([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            auto res = std::make_unique<server_task_result_slot_erase>();
            res->id       = id;
            res->n_erased = 7;
            queue_results.send(std::move(res));
        });
        auto f = std::async(std::launch::async, [&] { return rd.wait_for_all(connected); });
        auto br = wait_or_die(f, "a reader before the stop");
        sender.join();
        TCHECK(!br.is_terminated && br.error == nullptr && br.results.size() == 1 && br.results[0] != nullptr &&
               dynamic_cast<server_task_result_slot_erase *>(br.results[0].get()) != nullptr,
               "before the stop the reader must get the real result");
    }

    // 2. a non-streaming request and a streaming one wait; the stop wakes both with the error result at once
    server_response_reader rd_batch(queue_tasks, queue_results, 1);
    server_response_reader rd_stream(queue_tasks, queue_results, 1);
    post_metrics(rd_batch);
    post_metrics(rd_stream);
    auto f_batch  = std::async(std::launch::async, [&] { return rd_batch.wait_for_all(connected); });
    auto f_stream = std::async(std::launch::async, [&] { return rd_stream.next(connected); });
    std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // both readers past their first 1 s poll
    TCHECK(f_batch.wait_for(std::chrono::seconds(0)) != std::future_status::ready, "the non-streaming reader returned before the stop");
    TCHECK(f_stream.wait_for(std::chrono::seconds(0)) != std::future_status::ready, "the streaming reader returned before the stop");

    const auto t_stop = clk::now();
    queue_results.stop_waiters();
    auto br = wait_or_die(f_batch, "the non-streaming reader");
    const double ms_batch = ms_since(t_stop);
    auto rs = wait_or_die(f_stream, "the streaming reader");
    const double ms_stream = ms_since(t_stop);
    TCHECK(!br.is_terminated && is_stop_error(br.error), "the non-streaming reader must get the 503 shutdown error");
    TCHECK(is_stop_error(rs), "the streaming reader must get the 503 shutdown error");
    TCHECK(ms_batch < 500.0 && ms_stream < 500.0, "the readers took %.0f / %.0f ms after the stop (want < 500)", ms_batch, ms_stream);

    // 3. a request that starts after the stop does not wait either
    {
        server_response_reader rd(queue_tasks, queue_results, 1);
        post_metrics(rd);
        const auto t0 = clk::now();
        auto f = std::async(std::launch::async, [&] { return rd.wait_for_all(connected); });
        auto late = wait_or_die(f, "a reader that started after the stop");
        TCHECK(is_stop_error(late.error) && ms_since(t0) < 500.0, "a reader after the stop must get the error at once");
    }

    // 4. the blocking recv returns the error too
    {
        const int id = queue_tasks.get_new_id();
        queue_results.add_waiting_task_id(id);
        auto f = std::async(std::launch::async, [&] { return queue_results.recv(id); });
        auto r = wait_or_die(f, "a blocking recv");
        TCHECK(is_stop_error(r) && r->id == id, "a blocking recv after the stop must get the error for its task");
        queue_results.remove_waiting_task_id(id);
    }

    printf("test-server-stop: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
