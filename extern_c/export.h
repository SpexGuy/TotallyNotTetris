#ifndef _EXPORT_H
#define _EXPORT_H 1

#if defined _WIN32 || defined __CYGWIN__
    #define API __declspec(dllexport)
#else
    #define API
#endif

#if defined __cplusplus
    #define EXTERN extern "C"
#else
    #include <stdarg.h>
    #include <stdbool.h>
    #define EXTERN extern
#endif

#define ZIG_EXPORT EXTERN API

#endif
