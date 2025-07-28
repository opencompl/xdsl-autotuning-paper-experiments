#!/usr/bin/awk -f

BEGIN {
    MK = M * K
    KN = K * N
    MN = M * N
}

{
    # Basic dimension substitutions
    gsub(/{{M}}/, M)
    gsub(/{{N}}/, N)
    gsub(/{{K}}/, K)

    # Derived arithmetic expressions
    gsub(/{{MK}}/, MK)
    gsub(/{{KN}}/, KN)
    gsub(/{{MN}}/, MN)

    print
}
