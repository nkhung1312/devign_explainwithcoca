# D2A Leaderboard Data

This document describes D2A V1 Leaderboard data. You can download them from the Leaderboard section of the [D2A Dataset](https://dax-cdn.cdn.appdomain.cloud/dax-d2a/1.1.0/d2a.html?cm_mc_uid=52096571630515722723826&cm_mc_sid_50200000=65851751618339788874&_ga=2.42786284.851757668.1618339789-1229357178.1617837310) page. To begin download directly you can click [here](https://dax-cdn.cdn.appdomain.cloud/dax-d2a/1.1.0/d2a_leaderboard_data.tar.gz).

## Source files:

The files were created using the [default security errors](#default-security-errors) of datasets Libav, OpenSSL, Nginx, Httpd and Libtiff from [D2A](https://developer.ibm.com/exchanges/data/all/d2a/).

There are 4 directories corresponding to 4 tasks of the leaderboard. Each directory contains 3 csv files corresponding to the train (80%), dev (10%) and test (10%) split. 
The columns in the split files are identical except the test split which does not contain the label column.

## Columns:

1. **id**: A unique id for every example in a task.
2. **label**: Values are 0 or 1.
	1. Value 0: No vulnerability/defect in the example.
	2. Value 1: Example contains some vulnerability/defect.
3. **trace**: Bug trace or bug report generated by Infer static analyzer. Infer predictions contain a lot of False positives which is why even 0 label examples have a bug report. 
4. **bug_function/code**: Full source code of the function where the vulnerability originates.
5. **bug_url**: URL of the file which contains the bug_function.
6. **functions**: Full source code of all the functions in the bug trace, with the duplicates removed. This will include the function in bug_function.

## Default Security Errors:

These are security errors enabled by default by Infer.

* BIABD_USE_AFTER_FREE
* BUFFER_OVERRUN_L1
* BUFFER_OVERRUN_L2
* BUFFER_OVERRUN_L3
* BUFFER_OVERRUN_R2
* BUFFER_OVERRUN_S2
* BUFFER_OVERRUN_T1
* INTEGER_OVERFLOW_L1
* INTEGER_OVERFLOW_L2
* INTEGER_OVERFLOW_R2
* MEMORY_LEAK
* NULL_DEREFERENCE
* RESOURCE_LEAK
* LAB_RESOURCE_LEAK
* UNINITIALIZED_VALUE
* USE_AFTER_DELETE
* USE_AFTER_FREE
* USE_AFTER_LIFETIME

## Data Examples:

1. Trace:

```"test/bntest.c:1802: error: BUFFER_OVERRUN_L3
  Offset: [4, +oo] (‚áê [0, +oo] + 4) Size: [0, 8388607] by call to `BN_mul`.
Showing all 12 steps of the trace


test/bntest.c:1798:10: Call
1796.   
1797.       /* Test that BN_mul never gives negative zero. */
1798.       if (!BN_set_word(a, 1))
                 ^
1799.           goto err;
1800.       BN_set_negative(a, 1);

crypto/bn/bn_lib.c:463:1: Parameter `*a->d`
  461.   }
  462.   
  463. > int BN_set_word(BIGNUM *a, BN_ULONG w)
  464.   {
  465.       bn_check_top(a);

crypto/bn/bn_lib.c:466:9: Call
  464.   {
  465.       bn_check_top(a);
  466.       if (bn_expand(a, (int)sizeof(BN_ULONG) * 8) == NULL)
                 ^
  467.           return (0);
  468.       a->neg = 0;

crypto/bn/bn_lcl.h:676:1: Parameter `*a->d`
    674.   int bn_probable_prime_dh_coprime(BIGNUM *rnd, int bits, BN_CTX *ctx);
    675.   
    676. > static ossl_inline BIGNUM *bn_expand(BIGNUM *a, int bits)
    677.   {
    678.       if (bits > (INT_MAX - BN_BITS2 + 1))

test/bntest.c:1802:10: Call
1800.       BN_set_negative(a, 1);
1801.       BN_zero(b);
1802.       if (!BN_mul(c, a, b, ctx))
                 ^
1803.           goto err;
1804.       if (!BN_is_zero(c) || BN_is_negative(c)) {

crypto/bn/bn_mul.c:828:1: Parameter `*b->d`
  826.   #endif                          /* BN_RECURSION */
  827.   
  828. > int BN_mul(BIGNUM *r, const BIGNUM *a, const BIGNUM *b, BN_CTX *ctx)
  829.   {
  830.       int ret = 0;

crypto/bn/bn_mul.c:909:17: Call
  907.                   if (bn_wexpand(rr, k * 4) == NULL)
  908.                       goto err;
  909.                   bn_mul_part_recursive(rr->d, a->d, b->d,
                         ^
  910.                                         j, al - j, bl - j, t->d);
  911.               } else {            /* al <= j || bl <= j */

crypto/bn/bn_mul.c:480:1: Parameter `*b`
    478.    */
    479.   /* tnX may not be negative but less than n */
    480. > void bn_mul_part_recursive(BN_ULONG *r, BN_ULONG *a, BN_ULONG *b, int n,
    481.                              int tna, int tnb, BN_ULONG *t)
    482.   {

crypto/bn/bn_mul.c:488:9: Call
    486.   
    487.       if (n < 8) {
    488.           bn_mul_normal(r, a, n + tna, b, n + tnb);
                   ^
    489.           return;
    490.       }

crypto/bn/bn_mul.c:983:1: <Length trace>
981.   }
982.   
983. > void bn_mul_normal(BN_ULONG *r, BN_ULONG *a, int na, BN_ULONG *b, int nb)
984.   {
985.       BN_ULONG *rr;

crypto/bn/bn_mul.c:983:1: Parameter `*b`
      981.   }
      982.   
      983. > void bn_mul_normal(BN_ULONG *r, BN_ULONG *a, int na, BN_ULONG *b, int nb)
      984.   {
      985.       BN_ULONG *rr;

crypto/bn/bn_mul.c:1018:50: Array access: Offset: [4, +oo] (‚áê [0, +oo] + 4) Size: [0, 8388607] by call to `BN_mul` 
      1016.           if (--nb <= 0)
      1017.               return;
      1018.           rr[4] = bn_mul_add_words(&(r[4]), a, na, b[4]);
                                                               ^
      1019.           rr += 4;
      1020.           r += 4;
"
```

2. Bug URL:

```
https://github.com/openssl/openssl/blob/0282aeb690d63fab73a07191b63300a2fe30d212/crypto/bn/bn_mul.c/#L1018
```

3. Bug Function:

```
"void bn_mul_normal(BN_ULONG *r, BN_ULONG *a, int na, BN_ULONG *b, int nb)
{
    BN_ULONG *rr;
    if (na < nb) {
        int itmp;
        BN_ULONG *ltmp;
        itmp = na;
        na = nb;
        nb = itmp;
        ltmp = a;
        a = b;
        b = ltmp;
    }
    rr = &(r[na]);
    if (nb <= 0) {
        (void)bn_mul_words(r, a, na, 0);
        return;
    } else
        rr[0] = bn_mul_words(r, a, na, b[0]);
    for (;;) {
        if (--nb <= 0)
            return;
        rr[1] = bn_mul_add_words(&(r[1]), a, na, b[1]);
        if (--nb <= 0)
            return;
        rr[2] = bn_mul_add_words(&(r[2]), a, na, b[2]);
        if (--nb <= 0)
            return;
        rr[3] = bn_mul_add_words(&(r[3]), a, na, b[3]);
        if (--nb <= 0)
            return;
        rr[4] = bn_mul_add_words(&(r[4]), a, na, b[4]);
        rr += 4;
        r += 4;
        b += 4;
    }
}"
```

4. Functions:

```
[
'static int test_negzero() {
  BIGNUM * a = BN_new();
  BIGNUM * b = BN_new();
  BIGNUM * c = BN_new();
  BIGNUM * d = BN_new();
  BIGNUM * numerator = NULL, * denominator = NULL;
  int consttime, st = 0;
  if (a == NULL || b == NULL || c == NULL || d == NULL) goto err;
  if (!BN_set_word(a, 1)) goto err;
  BN_set_negative(a, 1);
  BN_zero(b);
  if (!BN_mul(c, a, b, ctx)) goto err;
  if (!BN_is_zero(c) || BN_is_negative(c)) {
    fprintf(stderr, "Multiplication test failed!");
    goto err;
  }
  for (consttime = 0; consttime < 2; consttime++) {
    numerator = BN_new();
    denominator = BN_new();
    if (numerator == NULL || denominator == NULL) goto err;
    if (consttime) {
      BN_set_flags(numerator, BN_FLG_CONSTTIME);
      BN_set_flags(denominator, BN_FLG_CONSTTIME);
    }
    if (!BN_set_word(numerator, 1) || !BN_set_word(denominator, 2)) goto err;
    BN_set_negative(numerator, 1);
    if (!BN_div(a, b, numerator, denominator, ctx)) goto err;
    if (!BN_is_zero(a) || BN_is_negative(a)) {
      fprintf(stderr, "Incorrect quotient (consttime = %d).", consttime);
      goto err;
    }
    if (!BN_set_word(denominator, 1)) goto err;
    if (!BN_div(a, b, numerator, denominator, ctx)) goto err;
    if (!BN_is_zero(b) || BN_is_negative(b)) {
      fprintf(stderr, "Incorrect remainder (consttime = %d).", consttime);
      goto err;
    }
    BN_free(numerator);
    BN_free(denominator);
    numerator = denominator = NULL;
  }
  BN_zero(a);
  BN_set_negative(a, 1);
  if (BN_is_negative(a)) {
    fprintf(stderr, "BN_set_negative produced a negative zero.");
    goto err;
  }
  st = 1;
  err: BN_free(a);
  BN_free(b);
  BN_free(c);
  BN_free(d);
  BN_free(numerator);
  BN_free(denominator);
  return st;
}', 
'int BN_set_word(BIGNUM * a, BN_ULONG w) {
  bn_check_top(a);
  if (bn_expand(a, (int) sizeof(BN_ULONG) * 8) == NULL) return (0);
  a -> neg = 0;
  a -> d[0] = w;
  a -> top = (w ? 1 : 0);
  bn_check_top(a);
  return (1);
}', 
'static ossl_inline BIGNUM * bn_expand(BIGNUM * a, int bits) {
  if (bits > (INT_MAX - BN_BITS2 + 1)) return NULL;
  if (((bits + BN_BITS2 - 1) / BN_BITS2) <= (a) -> dmax) return a;
  return bn_expand2((a), (bits + BN_BITS2 - 1) / BN_BITS2);
}', 
'int BN_mul(BIGNUM * r,
  const BIGNUM * a,
    const BIGNUM * b, BN_CTX * ctx) {
  int ret = 0;
  int top, al, bl;
  BIGNUM * rr;
  #if defined(BN_MUL_COMBA) || defined(BN_RECURSION) int i;
  #endif #ifdef BN_RECURSION BIGNUM * t = NULL;
  int j = 0, k;
  #endif bn_check_top(a);
  bn_check_top(b);
  bn_check_top(r);
  al = a -> top;
  bl = b -> top;
  if ((al == 0) || (bl == 0)) {
    BN_zero(r);
    return (1);
  }
  top = al + bl;
  BN_CTX_start(ctx);
  if ((r == a) || (r == b)) {
    if ((rr = BN_CTX_get(ctx)) == NULL) goto err;
  } else rr = r;
  rr -> neg = a -> neg ^ b -> neg;
  #if defined(BN_MUL_COMBA) || defined(BN_RECURSION) i = al - bl;
  #endif #ifdef BN_MUL_COMBA
  if (i == 0) {
    #
    if 0
    if (al == 4) {
      if (bn_wexpand(rr, 8) == NULL) goto err;
      rr -> top = 8;
      bn_mul_comba4(rr -> d, a -> d, b -> d);
      goto end;
    }
    # endif
    if (al == 8) {
      if (bn_wexpand(rr, 16) == NULL) goto err;
      rr -> top = 16;
      bn_mul_comba8(rr -> d, a -> d, b -> d);
      goto end;
    }
  }
  #endif #ifdef BN_RECURSION
  if ((al >= BN_MULL_SIZE_NORMAL) && (bl >= BN_MULL_SIZE_NORMAL)) {
    if (i >= -1 && i <= 1) {
      if (i >= 0) {
        j = BN_num_bits_word((BN_ULONG) al);
      }
      if (i == -1) {
        j = BN_num_bits_word((BN_ULONG) bl);
      }
      j = 1 << (j - 1);
      assert(j <= al || j <= bl);
      k = j + j;
      t = BN_CTX_get(ctx);
      if (t == NULL) goto err;
      if (al > j || bl > j) {
        if (bn_wexpand(t, k * 4) == NULL) goto err;
        if (bn_wexpand(rr, k * 4) == NULL) goto err;
        bn_mul_part_recursive(rr -> d, a -> d, b -> d, j, al - j, bl - j, t -> d);
      } else {
        if (bn_wexpand(t, k * 2) == NULL) goto err;
        if (bn_wexpand(rr, k * 2) == NULL) goto err;
        bn_mul_recursive(rr -> d, a -> d, b -> d, j, al - j, bl - j, t -> d);
      }
      rr -> top = top;
      goto end;
    }
    #
    if 0
    if (i == 1 && !BN_get_flags(b, BN_FLG_STATIC_DATA)) {
      BIGNUM * tmp_bn = (BIGNUM * ) b;
      if (bn_wexpand(tmp_bn, al) == NULL) goto err;
      tmp_bn -> d[bl] = 0;
      bl++;
      i--;
    } else if (i == -1 && !BN_get_flags(a, BN_FLG_STATIC_DATA)) {
      BIGNUM * tmp_bn = (BIGNUM * ) a;
      if (bn_wexpand(tmp_bn, bl) == NULL) goto err;
      tmp_bn -> d[al] = 0;
      al++;
      i++;
    }
    if (i == 0) {
      j = BN_num_bits_word((BN_ULONG) al);
      j = 1 << (j - 1);
      k = j + j;
      t = BN_CTX_get(ctx);
      if (al == j) {
        if (bn_wexpand(t, k * 2) == NULL) goto err;
        if (bn_wexpand(rr, k * 2) == NULL) goto err;
        bn_mul_recursive(rr -> d, a -> d, b -> d, al, t -> d);
      } else {
        if (bn_wexpand(t, k * 4) == NULL) goto err;
        if (bn_wexpand(rr, k * 4) == NULL) goto err;
        bn_mul_part_recursive(rr -> d, a -> d, b -> d, al - j, j, t -> d);
      }
      rr -> top = top;
      goto end;
    }
    # endif
  }
  #endif
  if (bn_wexpand(rr, top) == NULL) goto err;
  rr -> top = top;
  bn_mul_normal(rr -> d, a -> d, al, b -> d, bl);
  #if defined(BN_MUL_COMBA) || defined(BN_RECURSION) end: #endif bn_correct_top(rr);
  if (r != rr && BN_copy(r, rr) == NULL) goto err;
  ret = 1;
  err: bn_check_top(r);
  BN_CTX_end(ctx);
  return (ret);
}', 
'void bn_mul_part_recursive(BN_ULONG * r, BN_ULONG * a, BN_ULONG * b, int n, int tna, int tnb, BN_ULONG * t) {
  int i, j, n2 = n * 2;
  int c1, c2, neg;
  BN_ULONG ln, lo, * p;
  if (n < 8) {
    bn_mul_normal(r, a, n + tna, b, n + tnb);
    return;
  }
  c1 = bn_cmp_part_words(a, & (a[n]), tna, n - tna);
  c2 = bn_cmp_part_words( & (b[n]), b, tnb, tnb - n);
  neg = 0;
  switch (c1 * 3 + c2) {
  case -4:
    bn_sub_part_words(t, & (a[n]), a, tna, tna - n);
    bn_sub_part_words( & (t[n]), b, & (b[n]), tnb, n - tnb);
    break;
  case -3:
  case -2:
    bn_sub_part_words(t, & (a[n]), a, tna, tna - n);
    bn_sub_part_words( & (t[n]), & (b[n]), b, tnb, tnb - n);
    neg = 1;
    break;
  case -1:
  case 0:
  case 1:
  case 2:
    bn_sub_part_words(t, a, & (a[n]), tna, n - tna);
    bn_sub_part_words( & (t[n]), b, & (b[n]), tnb, n - tnb);
    neg = 1;
    break;
  case 3:
  case 4:
    bn_sub_part_words(t, a, & (a[n]), tna, n - tna);
    bn_sub_part_words( & (t[n]), & (b[n]), b, tnb, tnb - n);
    break;
  }
  #
  if 0
  if (n == 4) {
    bn_mul_comba4( & (t[n2]), t, & (t[n]));
    bn_mul_comba4(r, a, b);
    bn_mul_normal( & (r[n2]), & (a[n]), tn, & (b[n]), tn);
    memset( & r[n2 + tn * 2], 0, sizeof( * r) * (n2 - tn * 2));
  } else # endif
  if (n == 8) {
    bn_mul_comba8( & (t[n2]), t, & (t[n]));
    bn_mul_comba8(r, a, b);
    bn_mul_normal( & (r[n2]), & (a[n]), tna, & (b[n]), tnb);
    memset( & r[n2 + tna + tnb], 0, sizeof( * r) * (n2 - tna - tnb));
  } else {
    p = & (t[n2 * 2]);
    bn_mul_recursive( & (t[n2]), t, & (t[n]), n, 0, 0, p);
    bn_mul_recursive(r, a, b, n, 0, 0, p);
    i = n / 2;
    if (tna > tnb) j = tna - i;
    else j = tnb - i;
    if (j == 0) {
      bn_mul_recursive( & (r[n2]), & (a[n]), & (b[n]), i, tna - i, tnb - i, p);
      memset( & r[n2 + i * 2], 0, sizeof( * r) * (n2 - i * 2));
    } else if (j > 0) {
      bn_mul_part_recursive( & (r[n2]), & (a[n]), & (b[n]), i, tna - i, tnb - i, p);
      memset( & (r[n2 + tna + tnb]), 0, sizeof(BN_ULONG) * (n2 - tna - tnb));
    } else {
      memset( & r[n2], 0, sizeof( * r) * n2);
      if (tna < BN_MUL_RECURSIVE_SIZE_NORMAL && tnb < BN_MUL_RECURSIVE_SIZE_NORMAL) {
        bn_mul_normal( & (r[n2]), & (a[n]), tna, & (b[n]), tnb);
      } else {
        for (;;) {
          i /= 2;
          if (i < tna || i < tnb) {
            bn_mul_part_recursive( & (r[n2]), & (a[n]), & (b[n]), i, tna - i, tnb - i, p);
            break;
          } else if (i == tna || i == tnb) {
            bn_mul_recursive( & (r[n2]), & (a[n]), & (b[n]), i, tna - i, tnb - i, p);
            break;
          }
        }
      }
    }
  }
  c1 = (int)(bn_add_words(t, r, & (r[n2]), n2));
  if (neg) {
    c1 -= (int)(bn_sub_words( & (t[n2]), t, & (t[n2]), n2));
  } else {
    c1 += (int)(bn_add_words( & (t[n2]), & (t[n2]), t, n2));
  }
  c1 += (int)(bn_add_words( & (r[n]), & (r[n]), & (t[n2]), n2));
  if (c1) {
    p = & (r[n + n2]);
    lo = * p;
    ln = (lo + c1) & BN_MASK2;* p = ln;
    if (ln < (BN_ULONG) c1) {
      do {
        p++;
        lo = * p;
        ln = (lo + 1) & BN_MASK2;* p = ln;
      } while (ln == 0);
    }
  }
}', 
'void bn_mul_normal(BN_ULONG * r, BN_ULONG * a, int na, BN_ULONG * b, int nb) {
  BN_ULONG * rr;
  if (na < nb) {
    int itmp;
    BN_ULONG * ltmp;
    itmp = na;
    na = nb;
    nb = itmp;
    ltmp = a;
    a = b;
    b = ltmp;
  }
  rr = & (r[na]);
  if (nb <= 0) {
    (void) bn_mul_words(r, a, na, 0);
    return;
  } else rr[0] = bn_mul_words(r, a, na, b[0]);
  for (;;) {
    if (--nb <= 0) return;
    rr[1] = bn_mul_add_words( & (r[1]), a, na, b[1]);
    if (--nb <= 0) return;
    rr[2] = bn_mul_add_words( & (r[2]), a, na, b[2]);
    if (--nb <= 0) return;
    rr[3] = bn_mul_add_words( & (r[3]), a, na, b[3]);
    if (--nb <= 0) return;
    rr[4] = bn_mul_add_words( & (r[4]), a, na, b[4]);
    rr += 4;
    r += 4;
    b += 4;
  }
}'
]
```

[Leaderboard README](https://github.com/IBM/D2A/blob/main/leaderboard/README.md) || [Leaderboard page](https://ibm.github.io/D2A)
