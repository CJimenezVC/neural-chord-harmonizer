// Minimal test harness — no external dependencies.
#include "test_framework.h"

int avt::TestRegistry::failures = 0;

int main()
{
    avt::TestRegistry::runAll();
    return avt::TestRegistry::failures == 0 ? 0 : 1;
}
