#include <iostream>
#include <thread>
// Use #include "task_thread_pool.hpp" for relative path,
// and #include <task_thread_pool.hpp> if installed in include path
#include "task_thread_pool.hpp"

// func outside class with arguments input and return value
bool outerFunc1(const bool& arg1, const bool& arg2) {
    bool arg = arg1 && arg2;
    std::cout << "call to outerFunc1: " << (arg) << std::endl;
    return arg;
}

// func outside class with return value
bool outerFunc2() {
    bool arg = true;
    std::cout << "call to outerFunc2: " << (arg) << std::endl;
    return arg;
}

// func outside class with arguments input
void outerFunc3(const bool& arg1, const bool& arg2) {
    bool arg = arg1 && arg2;
    std::cout << "call to outerFunc3: " << (arg) << std::endl;
    return;
}

// func outside class
void outerFunc4() {
    bool arg = false;
    std::cout << "call to outerFunc4: " << (arg) << std::endl;
    return;
}

class TestClass {
public:
    int cnt;
    TestClass() {
        cnt = 0;
    }
    void add() {
        cnt ++;
    }
};

class ThreadDemo {
private:
    static task_thread_pool::task_thread_pool pool_;

    // func in class should be "static"
    // func inside class with arguments input and return value
    static bool innerFunc1(const bool& arg1, const bool& arg2) {
        bool arg = arg1 && arg2;
        std::cout << "call to innerFunc1: " << (arg) << std::endl;
        return arg;
    }

    // func inside class with return value
    static bool innerFunc2() {
        bool arg = true;
        std::cout << "call to innerFunc2: " << (arg) << std::endl;
        return arg;
    }

    // func inside class with arguments input
    static void innerFunc3(const bool& arg1, const bool& arg2) {
        bool arg = arg1 && arg2;
        std::cout << "call to innerFunc3: " << (arg) << std::endl;
        return;
    }

    // func inside class
    static void innerFunc4() {
        bool arg = false;
        std::cout << "call to innerFunc4: " << (arg) << std::endl;
        return;
    }

    static void testFunc(TestClass*& test) {
        int cnt = 0;
        while (cnt++ < 100) {
            (*test).add();
            //break;
        }
    }

    static void monitor(TestClass*& test) {
        int cnt = 0;
        while (cnt++ < 100) {
            std::cout << (*test).cnt << std::endl;
            //break;
        }
    }

public:
    void thread_test() {
        std::future<bool> func_future_1, func_future_2, func_future_3, func_future_4;
        func_future_1 = pool_.submit(outerFunc1, true, true);
        func_future_2 = pool_.submit(outerFunc2);
        pool_.submit_detach(outerFunc3, true, false);
        pool_.submit_detach(outerFunc4);
        func_future_3 = pool_.submit(innerFunc1, true, true);
        func_future_4 = pool_.submit(innerFunc2);
        pool_.submit_detach(innerFunc3, true, false);
        pool_.submit_detach(innerFunc4);

        pool_.wait_for_tasks();
    }

    void test(TestClass* test1) {
        pool_.pause();
        pool_.submit_detach(testFunc, test1);
        pool_.submit_detach(monitor, test1);
        pool_.submit_detach(testFunc, test1);
        pool_.submit_detach(monitor, test1);
        pool_.unpause();
        // pool_.submit_detach(testFunc, test1);
        // monitor();
        // testFunc(test1);
        // monitor(test1);     
    }
};

task_thread_pool::task_thread_pool ThreadDemo::pool_;

int main() {
    ThreadDemo demo;
    TestClass* test1 = new TestClass();
    // demo.thread_test();
    demo.test(test1);
    return 0;
}