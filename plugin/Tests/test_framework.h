#pragma once

// Tiny zero-dependency test framework: TEST_CASE registers a function;
// CHECK records failures. Good enough for the DSP primitive smoke tests.

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace avt
{
    struct TestRegistry
    {
        static std::vector<std::pair<std::string, std::function<void()>>>& cases()
        {
            static std::vector<std::pair<std::string, std::function<void()>>> c;
            return c;
        }

        static int failures;

        static void add (const std::string& name, std::function<void()> fn)
        {
            cases().emplace_back (name, std::move (fn));
        }

        static void runAll()
        {
            for (auto& [name, fn] : cases())
            {
                std::printf ("[ RUN  ] %s\n", name.c_str());
                fn();
                std::printf ("[ OK   ] %s\n", name.c_str());
            }
            std::printf ("\n%zu tests, %d failures\n", cases().size(), failures);
        }
    };

    struct AutoReg
    {
        AutoReg (const std::string& name, std::function<void()> fn)
        {
            TestRegistry::add (name, std::move (fn));
        }
    };
}

#define TEST_CASE(name)                                                        \
    static void name();                                                        \
    static avt::AutoReg reg_##name { #name, name };                            \
    static void name()

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (! (cond)) {                                                        \
            std::printf ("[ FAIL ] %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++avt::TestRegistry::failures;                                     \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, tol)  CHECK (std::fabs ((a) - (b)) <= (tol))
