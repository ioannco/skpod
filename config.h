#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define MINI_DATASET
#endif

#ifdef MINI_DATASET
#define N (2*2*2*2 + 2)
#ifndef CHECKSUM
#define CHECKSUM 9202.905273
#endif
#endif

#ifdef SMALL_DATASET
#define N (2*2*2*2*2 + 2)
#ifndef CHECKSUM
#define CHECKSUM 130680.328125
#endif
#endif

#ifdef MEDIUM_DATASET
#define N (2*2*2*2*2*2 + 2)
#ifndef CHECKSUM
#define CHECKSUM 2723874.750000
#endif
#endif

#ifdef LARGE_DATASET
#define  N  (2*2*2*2*2*2*2 + 2)
#ifndef CHECKSUM
#define CHECKSUM 54117800.000000
#endif
#endif

#ifdef EXTRA_LARGE_DATASET
#define  N  (2*2*2*2*2*2*2*2 + 2)
#ifndef CHECKSUM
#define CHECKSUM 953108160.000000
#endif
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif