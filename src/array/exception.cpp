#include <string.h> // strncpy
#include <stdio.h>
#include <af/exception.h>

#ifdef OS_WIN
#define snprintf _snprintf
#endif

namespace af {

exception::exception()
{
    strncpy(m_msg, "unknown exception", sizeof(m_msg));
}

exception::exception(const char *msg)
{
    strncpy(m_msg, msg, sizeof(m_msg));
    m_msg[sizeof(m_msg)-1] = '\0';
}

exception::exception(const char *file, unsigned line)
{
    snprintf(m_msg, sizeof(m_msg)-1, "%s:%d: exception thrown", file, line);
    m_msg[sizeof(m_msg)-1] = '\0';
}

exception::exception(const char *file, unsigned line, af_err err)
{
    snprintf(m_msg, sizeof(m_msg)-1, "%s:%d: AF_ERROR %d", file, line, (int)(err));
    m_msg[sizeof(m_msg)-1] = '\0';
}

}
