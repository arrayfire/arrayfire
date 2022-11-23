// Umar Arshad
// Copyright 2014

// this enables template overloads of standard CRT functions that call the
// more secure variants automatically,
#define _CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES 1

#include <cstring>
// strtok symbol name that keeps context is not on windows and linux
// so, the above overload define won't help with that function
#if defined(OS_WIN)
#define STRTOK_CALL(...) strtok_s(__VA_ARGS__)
#else
#define STRTOK_CALL(...) strtok_r(__VA_ARGS__)
#endif

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>  // IWYU pragma: keep
#include <string>
#include <utility>
#include <vector>

#include <common/deterministicHash.hpp>

using namespace std;
using std::cout;
typedef map<string, string> opt_t;

void print_usage() {
    cout << R"delimiter(BIN2CPP
Converts files from a binary file to C++ headers. It is similar to bin2c and
xxd but adds support for namespaces.

| --name        | name of the variable (default: var)                               |
| --file        | input file                                                        |
| --output      | output file (If no output is specified then it prints to stdout   |
| --type        | Type of variable (default: char)                                  |
| --binary      | If the file contents are in binary form                           |
| --nullterm    | Add a null character to the end of the file                       |
| --namespace   | A space seperated list of namespaces                              |
| --formatted   | Tabs for formatting                                               |
| --version     | Prints my name                                                    |
| --help        | Prints usage info                                                 |

Example
-------
Command:
./bin2cpp --file blah.txt --namespace blah detail --formatted --name blah_var

Will produce:
#pragma once
#include <common/util.hpp>
#include <cstddef>
namespace blah {
	namespace detail {
		static const unsigned char blah_var_uchar [] = {
			0x2f,	0x2f,	0x20,	0x62,	0x6c,	0x61,	0x68,	0x2e,	0x74,	0x78,
			0x74,	0xa,	0x62,	0x6c,	0x61,	0x68,	0x20,	0x62,	0x6c,	0x61,
			0x68,	0x20,	0x62,	0x6c,	0x61,	0x68,	0xa,	};
		static const char *blah_var = (const char*)blah_var_uchar;
		static const size_t blah_var_len  = 27;
		static const size_t blah_var_hash = 12345678901234567890ULL;
		static const common::Source blah_var_src = {
			blah_var,
			blah_var_len,
			blah_var_hash
		};
	}
})delimiter";
    exit(0);
}

static bool formatted;
static bool binary   = false;
static bool nullterm = false;

void add_tabs(const int level) {
    if (formatted) {
        for (int i = 0; i < level; i++) { cout << "\t"; }
    }
}

opt_t parse_options(const vector<string> &args) {
    opt_t options;

    options["--name"]      = "";
    options["--type"]      = "";
    options["--file"]      = "";
    options["--output"]    = "";
    options["--namespace"] = "";

    // Parse Arguments
    string curr_opt;
    bool verbose = false;
    for (auto arg : args) {
        if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--binary") {
            binary = true;
        } else if (arg == "--nullterm") {
            nullterm = true;
        } else if (arg == "--formatted") {
            formatted = true;
        } else if (arg == "--version") {
            cout << args[0] << " By Umar Arshad" << endl;
        } else if (arg == "--help") {
            print_usage();
        } else if (options.find(arg) != options.end()) {
            curr_opt = arg;
        } else if (curr_opt.empty()) {
            // cerr << "Invalid Argument: " << arg << endl;
        } else {
            if (options[curr_opt] != "") {
                options[curr_opt] += " " + arg;
            } else {
                options[curr_opt] += arg;
            }
        }
    }

    if (verbose) {
        for (auto opts : options) {
            cout << get<0>(opts) << " " << get<1>(opts) << endl;
        }
    }
    return options;
}

stringstream removeComments(ifstream &input, string &filename) {
    stringstream ss;
    char line[256]{
        '\0'};  // Maximum length of lines in OpenCL code is limited to 256
    const char *tokenCommentsStart = "/*";
    const char *tokenCommentsEnd   = "*/";
    const char *tokenCommentsLine  = "//";
    const char *tokenString        = "\"";
    const char *delimitors         = " \t;";  // Only the subset we need
    enum { NO, STRING, ENDOFLINE, MULTILINE } commentsLevel{NO};

    while (input.getline(line, sizeof(line) - 1)) {
        char local[sizeof(line)];
        struct segment {
            char *start;
            char *end;
        } del{commentsLevel == MULTILINE ? line : nullptr, nullptr};
        vector<segment> dels;
        memcpy(local, line, sizeof(line));   // will be overwritten by strtok
        local[sizeof(local) - 1] = '\0';     // string is always terminated
        char *context            = nullptr;
        char *token              = STRTOK_CALL(local, delimitors, &context);
        do {
            char *subtoken = nullptr;
            while (token) {
                switch (commentsLevel) {
                    case MULTILINE:
                        subtoken = strstr(token, tokenCommentsEnd);
                        if (subtoken != nullptr) {
                            if (del.start == nullptr) del.start = line;
                            del.end = subtoken + strlen(tokenCommentsEnd) -
                                      local + line;
                            dels.push_back(del);
                            del           = {nullptr, nullptr};
                            token         = subtoken + strlen(tokenCommentsEnd);
                            commentsLevel = NO;
                        } else {
                            token = nullptr;
                        }
                        break;
                    case STRING:
                        subtoken = strstr(token, tokenString);
                        if (subtoken != nullptr) {
                            token         = subtoken + strlen(tokenString);
                            commentsLevel = NO;
                        } else {
                            token = nullptr;
                        }
                        break;
                    case NO: {
                        // select first subtoken inside this token
                        subtoken = strstr(token, tokenCommentsStart);
                        if (subtoken != nullptr) { commentsLevel = MULTILINE; }
                        char *ptr = strstr(token, tokenCommentsLine);
                        if ((ptr != nullptr) &&
                            ((subtoken == nullptr) || (ptr < subtoken))) {
                            commentsLevel = ENDOFLINE;
                            subtoken      = ptr;
                        }
                        ptr = strstr(token, tokenString);
                        if ((ptr != nullptr) &&
                            ((subtoken == nullptr) || ptr < subtoken)) {
                            commentsLevel = STRING;
                            subtoken      = ptr;
                        }
                        switch (commentsLevel) {
                            case MULTILINE:
                                del.start = subtoken - local + line;
                                token = subtoken + strlen(tokenCommentsStart);
                                break;
                            case ENDOFLINE:
                                del.start = subtoken - local + line;
                                token = subtoken + strlen(tokenCommentsLine);
                                break;
                            case STRING:
                                token = subtoken + strlen(tokenString);
                                break;
                            case NO:
                            default: token = nullptr;
                        }
                    } break;
                    case ENDOFLINE:
                    default: token = nullptr;
                }
            }
            token = STRTOK_CALL(nullptr, delimitors, &context);
        } while (token != nullptr);
        if (del.start != nullptr) {
            if (commentsLevel == ENDOFLINE) commentsLevel = NO;
            del.end = line + strlen(line);
            dels.push_back(del);
            del = {nullptr, nullptr};
        }
        // Delete all segments starting from the end!!!
        for (auto d = dels.crbegin(); d != dels.crend(); d++) {
            char *ptr1 = d->start;
            char *ptr2 = d->end;
            // Do not use strncpy, it has problems with overlapping because the
            // order isn't defined in the standard
            while ((*ptr2 != '\0') && (ptr2 != line + sizeof(line))) { *ptr1++ = *ptr2++; }
            *ptr1 = '\0';
        }
        // Remove trailing blanks
        for (long i = static_cast<long>(std::min(sizeof(line),strlen(line))) - 1;
             (i >= 0) && (line[i] == ' '); --i) {
            line[i] = '\0';
        }
        // Remove leading blanks
        char *linePtr = line;
        for (size_t i = 0, len = std::min(sizeof(line),strlen(line));
            (i < len) && (line[i] == ' ');
             ++i, ++linePtr) {}
        // Useful text is terminated by '\n';
        if (linePtr[0] != '\0') { ss << linePtr << "\n"; }
    }
    return (ss);
}

int main(int argc, const char *const *const argv) {
    vector<string> args(argv, argv + argc);

    if (argc == 1) {
        print_usage();
        return 0;
    }
    opt_t &&options = parse_options(args);

    // Save default cout buffer. Need this to prevent crash.
    auto bak = cout.rdbuf();
    unique_ptr<ofstream> outfile;

    // Set defaults
    if (options["--name"] == "") { options["--name"] = "var"; }
    if (options["--output"] != "") {
        // redirect stream if output file is specified
        outfile.reset(new ofstream(options["--output"]));
        cout.rdbuf(outfile->rdbuf());
    }

    cout << "#pragma once\n";
    cout << "#include <cstddef>\n";          // defines size_t
    cout << "#include <common/Source.hpp>\n";  // defines common::Source

    int ns_cnt = 0;
    int level  = 0;
    if (options["--namespace"] != "") {
        stringstream namespaces(options["--namespace"]);
        string name;
        namespaces >> name;
        do {
            add_tabs(level++);
            cout << "namespace " << name << " { \n";
            ns_cnt++;
            namespaces >> name;
        } while (!namespaces.fail());
    }

    if (options["--type"] == "") { options["--type"] = "char"; }
    add_tabs(level);

    // Always create unsigned char to avoid narrowing
    cout << "static const "
         << "unsigned char"
         << " " << options["--name"] << "_uchar [] = {\n";

    ifstream input(options["--file"],
                   (binary ? std::ios::binary : std::ios::in));
    size_t char_cnt = 0;
    stringstream ss = removeComments(input, options["--file"]);
    add_tabs(++level);
    for (char i; ss.get(i);) {
        cout << "0x" << std::hex << static_cast<int>(i & 0xff) << ",\t";
        char_cnt++;
        if (!(char_cnt % 10)) {
            cout << endl;
            add_tabs(level);
        }
    }

    if (nullterm) {
        // Add end of file character
        cout << "0x0";
        char_cnt++;
    }

    cout << "};\n";
    add_tabs(--level);

    // Cast to proper output type
    cout << "static const " << options["--type"] << " *" << options["--name"]
         << " = (const " << options["--type"] << " *)" << options["--name"]
         << "_uchar;\n";
    add_tabs(level);
    cout << "static const size_t " << options["--name"] << "_len"
         << "  = " << std::dec << char_cnt << ";\n";
    add_tabs(level);
    cout << "static const size_t " << options["--name"] << "_hash"
         << " = " << deterministicHash(ss.str()) << "ULL;\n";
    add_tabs(level);
    cout << "static const common::Source " << options["--name"] << "_src{\n";
    add_tabs(++level);
    cout << options["--name"] << ",\n";
    add_tabs(level);
    cout << options["--name"] << "_len,\n";
    add_tabs(level);
    cout << options["--name"] << "_hash\n";
    add_tabs(--level);
    cout << "};\n";

    while (ns_cnt--) {
        add_tabs(--level);
        cout << "}\n";
    }

    cout.rdbuf(bak);

    return 0;
}
