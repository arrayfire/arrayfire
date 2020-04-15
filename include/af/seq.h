/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

/**
    \struct af_seq

    \brief C-style struct to creating sequences for indexing

    \ingroup index_mat
*/
typedef struct af_seq {
    /// Start position of the sequence
    double begin;

    /// End position of the sequence (inclusive)
    double end;

    /// Step size between sequence values
    double step;
} af_seq;

static const af_seq af_span = {1, 1, 0};

#ifdef __cplusplus
namespace af
{
class array;

/**
    \class seq

    \brief seq is used to create sequences for indexing af::array

    \ingroup arrayfire_class
*/
class AFAPI seq
{
public:
    ///
    /// \brief Get the \ref af_seq C-style struct
    ///
    af_seq s;

    ///
    /// \brief Get's the length of the sequence
    ///
    size_t size;

    ///
    /// \brief Flag for gfor
    ///
    bool m_gfor;

    /**
        \brief Creates a sequence of size length as [0, 1, 2..., length - 1]

        The sequence has begin as 0, end as length - 1 and step as 1.

        \note When doing seq(-n), where n is > 0, then the sequence is generated as
        0...-n but step remains +1. This is because when such a sequence is
        used for indexing af::array, then -n represents n elements from the
        end. That is, seq(-2) will imply indexing an array 0...dimSize - 2.

        \code
                            // [begin, end, step]
        seq a(10);          // [0, 9, 1]    => 0, 1, 2....9
        \endcode

        \param[in] length is the size of the seq to be created.
    */
    seq(double length = 0);

    /**
        \brief Destructor
    */
    ~seq();

    /**
        \brief Creates a sequence starting at begin,
        ending at or before end (inclusive) with increments as step.

        The sequence will be [begin, begin + step, begin + 2 * step...., begin + n * step]
        where the begin + n * step <= end.

        \code
                            // [begin, end, step]
        seq a(10, 20);      // [10, 20, 1]  => 10, 11, 12....20
        seq b(10, 20, 2);   // [10, 20, 2]  => 10, 12, 14....20
        seq c(-5, 5);       // [-5, 5, 1]   => -5, -4, -3....0, 1....5
        seq d(-5, -15, -1); // [-5,-15, -1] => -5, -6, -7....-15
        seq e(-15, -5, 1);  // [-15, -5, 1] => -15, -14, -13....-5
        \endcode

        \param[in] begin is the start of the sequence
        \param[in] end is the maximum value a sequence can take (inclusive)
        \param[in] step is the increment or decrement size (default is 1)
    */
    seq(double begin, double end, double step = 1);

    /**
        \brief Copy constructor

        Creates a copy seq from another sequence.

        \param[in] other seqence to be copies
        \param[in] is_gfor is the gfor flag
    */
    seq(seq other, bool is_gfor);

    /**
        \brief Create a seq object from an \ref af_seq struct

        \param[in] s_ is the \ref af_seq struct
    */
    seq(const af_seq& s_);

    /**
        \brief Assignment operator to create a new sequence from an af_seq

        This operator creates a new sequence using the begin, end and step
        from the input sequence.

        \param[in] s is the input sequence
    */
    seq& operator=(const af_seq& s);

    /**
        \brief Negation operator creates a sequence with the signs negated

        begin is changed to -begin
        end is changed to -end
        step is changed to -step

        \code
                        // [begin, end, step]
        seq a(1, 10);   // [ 1, 10, 1] => 1, 2, 3....10
        seq b = -a;     // [-1,-10,-1] => -1, -2, -3...-10
        \endcode
    */
    inline seq operator-()         { return seq(-s.begin, -s.end,  -s.step); }

    /**
        \brief Addition operator offsets the begin and end by x. There is no
        change in step.

        begin is changed to begin + x
        end is changed to end + x

        \code
                            // [begin, end, step]
        seq a(2, 20, 2);    // [2, 20, 2] => 2, 4, 6....20
        seq b = a + 3;      // [5, 23, 2] => 5, 7, 9....23
        \endcode
    */
    inline seq operator+(double x) { return seq(s.begin + x, s.end + x, s.step); }

    /**
        \brief Subtraction operator offsets the begin and end by x. There is no
        change in step.

        begin is changed to begin - x
        end is changed to end - x

        \code
                            // [begin, end, step]
        seq a(10, 20, 2);   // [10, 20, 2] => 10, 12, 14....20
        seq b(2, 10);       // [ 2, 10, 1] => 2, 3, 4....10
        seq c = a - 3;      // [ 7, 17, 2] => 7, 9, 11....17
        seq d = b - 3;      // [-1,  7, 2] => -1, 1, 3....7
        \endcode
    */
    inline seq operator-(double x) { return seq(s.begin - x, s.end - x, s.step); }

    /**
        \brief Multiplication operator spaces the sequence by a factor x.

        begin is changed to begin * x
        end is changed to end * x
        step is changed to step * x

        \code
                            // [begin, end, step]
        seq a(10, 20, 2);   // [10, 20, 2] => 10, 12, 14....20
        seq b(-5, 5);       // [-5, 5, 1] => -5, -4, -3....0, 1....5
        seq c = a * 3;      // [30, 60, 6] => 30, 36, 42....60
        seq d = b * 3;      // [-15, 15, 3] => -15, -12, -9....0, 3....15
        seq e = a * 0.5;    // [5, 10, 1] => 5, 6, 7....10
        \endcode
    */
    inline seq operator*(double x) { return seq(s.begin * x, s.end * x, s.step * x); }

    friend inline seq operator+(double x, seq y) { return  y + x; }

    friend inline seq operator-(double x, seq y) { return -y + x; }

    friend inline seq operator*(double x, seq y) { return  y * x; }

    /**
        \brief Implicit conversion operator from seq to af::array

        Convertes a seq object into an af::array object. The contents of the
        af:array will be the explicit values from the seq.

        \note Do not use this to create arrays of sequences. Use \ref range.

        \code
                            // [begin, end, step]
        seq s(10, 20, 2);   // [10, 20, 2] => 10, 12, 14....20
        array arr = s;
        af_print(arr);      // 10    12    14    16    18    20
        \endcode
    */
    operator array() const;

    private:
    void init(double begin, double end, double step);
};

/// A special value representing the last value of an axis
extern AFAPI int end;

/// A special value representing the entire axis of an af::array
extern AFAPI seq span;

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Create a new af_seq object.
AFAPI af_seq af_make_seq(double begin, double end, double step);

#ifdef __cplusplus
}
#endif
