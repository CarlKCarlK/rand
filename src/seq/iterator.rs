// Copyright 2018-2024 Developers of the Rand project.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! `IteratorRandom`

#[allow(unused)]
use super::IndexedRandom;
use crate::{Rng, RngExt};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Equivalent to [`Iterator::nth`], but accepts a zero-based `u128` index.
///
/// An index which fits in `usize` requires one call to [`Iterator::nth`].
/// Larger indices are handled by repeatedly calling `nth(usize::MAX)`, each
/// time discarding `usize::MAX + 1` items, before retrieving the target item.
#[inline]
fn nth_u128<I: Iterator>(iter: &mut I, mut n: u128) -> Option<I::Item> {
    const USIZE_MAX: u128 = usize::MAX as u128;
    const CHUNK: u128 = USIZE_MAX + 1;

    // This branch is usually easy to predict: for nearly all calls it is
    // immediately false; for larger indices it remains true until the final chunk.
    while n > USIZE_MAX {
        iter.nth(usize::MAX)?;
        n -= CHUNK;
    }

    iter.nth(n as usize)
}

/// Extension trait on iterators, providing random sampling methods.
///
/// This trait is implemented on all iterators `I` where `I: Iterator + Sized`
/// and provides methods for
/// choosing one or more elements. You must `use` this trait:
///
/// ```
/// use rand::seq::IteratorRandom;
///
/// let faces = "😀😎😐😕😠😢";
/// println!("I am {}!", faces.chars().choose(&mut rand::rng()).unwrap());
/// ```
/// Example output (non-deterministic):
/// ```none
/// I am 😀!
/// ```
pub trait IteratorRandom: Iterator + Sized {
    /// Uniformly sample one element
    ///
    /// Assuming that the [`Iterator::size_hint`] is correct, this method
    /// returns one uniformly-sampled random element of the iterator, or `None`
    /// only if the iterator is empty. Incorrect bounds on the `size_hint` may
    /// cause this method to incorrectly return `None` if fewer elements than
    /// the advertised `lower` bound are present and may prevent sampling of
    /// elements beyond an advertised `upper` bound (i.e. incorrect `size_hint`
    /// is memory-safe, but may result in unexpected `None` result and
    /// non-uniform distribution).
    ///
    /// With an accurate [`Iterator::size_hint`] and where [`Iterator::nth`] is
    /// a constant-time operation, this method can offer `O(1)` performance.
    /// Where the exact size is unavailable, this uses skip-based reservoir
    /// sampling. Complexity is `O(n)` where `n` is the iterator length when
    /// [`Iterator::nth`] is linear, but may be significantly better when
    /// `nth` can skip elements efficiently.
    ///
    /// Note further that [`Iterator::size_hint`] may affect the number of RNG
    /// samples used as well as the result (while remaining uniform sampling).
    /// Consider instead using [`IteratorRandom::choose_stable`] to avoid
    /// [`Iterator`] combinators which only change size hints from affecting the
    /// results.
    ///
    /// # Panics
    ///
    /// For an iterator without an exact size, panics if selecting an element
    /// would require consuming more than `u64::MAX` elements.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::seq::IteratorRandom;
    ///
    /// let words = "Mary had a little lamb".split(' ');
    /// println!("{}", words.choose(&mut rand::rng()).unwrap());
    /// ```
    fn choose<R>(mut self, rng: &mut R) -> Option<Self::Item>
    where
        R: Rng + ?Sized,
    {
        let (lower, upper) = self.size_hint();

        // Handling for this condition outside the loop allows the optimizer to eliminate the loop
        // when the Iterator is an ExactSizeIterator. This has a large performance impact on e.g.
        // seq_iter_choose_from_1000.
        if upper == Some(lower) {
            match lower {
                0 => None,
                1 => self.next(),
                _ => self.nth(rng.random_range(..lower)),
            }
        } else {
            self.choose_stable(rng)
        }
    }

    /// Uniformly sample one element (stable)
    ///
    /// This method is very similar to [`choose`] except that the selected index
    /// only depends on the length of the iterator and the values produced by
    /// `rng`. Notably, for any iterator of a given length, this will make the
    /// same requests to `rng` and, if `rng` produces the same sequence of
    /// values, will select the same index from `self`. This may be useful if you
    /// need consistent results no matter what type of iterator you are working
    /// with. If you do not need this stability, prefer [`choose`].
    ///
    /// This method makes `O(log n)` calls to `rng` in expectation, where `n`
    /// is the iterator length.
    ///
    /// This method may use [`Iterator::nth`] to efficiently skip elements
    /// which cannot be selected.
    ///
    /// # Panics
    ///
    /// Panics if selecting an element would require consuming more than
    /// `u64::MAX` elements.
    ///
    /// [`choose`]: IteratorRandom::choose
    fn choose_stable<R>(mut self, rng: &mut R) -> Option<Self::Item>
    where
        R: Rng + ?Sized,
    {
        let mut result = self.next()?;
        let mut consumed = 1u128;

        // K=1 case of the skip method from Park et al. (2004):
        // "Reservoir-based Random Sampling with Replacement from Data Stream".
        // https://doi.org/10.1137/1.9781611972740.53
        const SCALE: u128 = 1 << 64;
        const MAX_POSITION: u128 = u64::MAX as u128;

        loop {
            // Sample r uniformly from 2^63 equally spaced points in (0, 1),
            // represented exactly as `numerator / SCALE`.
            let numerator = u128::from(rng.next_u64() >> 1) * 2 + 1;

            // ceil(r * consumed / (1 - r))
            // = ceil(consumed * SCALE / denominator) - consumed, where
            // denominator = SCALE - numerator represents (1 - r).
            let denominator = SCALE - numerator;
            let distance = (consumed * SCALE).div_ceil(denominator) - consumed;
            debug_assert_ne!(distance, 0);

            let Some(new_result) = nth_u128(&mut self, distance - 1) else {
                return Some(result);
            };
            result = new_result;
            consumed += distance;
            assert!(
                consumed <= MAX_POSITION,
                "selecting an element would require consuming more than u64::MAX elements"
            );
        }
    }

    /// Uniformly sample `amount` distinct elements into a buffer
    ///
    /// Collects values at random from the iterator into a supplied buffer
    /// until that buffer is filled.
    ///
    /// Although the elements are selected randomly, the order of elements in
    /// the buffer is neither stable nor fully random. If random ordering is
    /// desired, shuffle the result.
    ///
    /// Returns the number of elements added to the buffer. This equals the length
    /// of the buffer unless the iterator contains insufficient elements, in which
    /// case this equals the number of elements available.
    ///
    /// Complexity is `O(n)` where `n` is the length of the iterator.
    /// For slices, prefer [`IndexedRandom::sample`].
    fn sample_fill<R>(mut self, rng: &mut R, buf: &mut [Self::Item]) -> usize
    where
        R: Rng + ?Sized,
    {
        let amount = buf.len();
        let mut len = 0;
        while len < amount {
            if let Some(elem) = self.next() {
                buf[len] = elem;
                len += 1;
            } else {
                // Iterator exhausted; stop early
                return len;
            }
        }

        // Continue, since the iterator was not exhausted
        for (i, elem) in self.enumerate() {
            let k = rng.random_range(..i + 1 + amount);
            if let Some(slot) = buf.get_mut(k) {
                *slot = elem;
            }
        }
        len
    }

    /// Uniformly sample `amount` distinct elements into a [`Vec`]
    ///
    /// This is equivalent to `sample_fill` except for the result type.
    ///
    /// Although the elements are selected randomly, the order of elements in
    /// the buffer is neither stable nor fully random. If random ordering is
    /// desired, shuffle the result.
    ///
    /// The length of the returned vector equals `amount` unless the iterator
    /// contains insufficient elements, in which case it equals the number of
    /// elements available.
    ///
    /// Complexity is `O(n)` where `n` is the length of the iterator.
    /// For slices, prefer [`IndexedRandom::sample`].
    #[cfg(feature = "alloc")]
    fn sample<R>(mut self, rng: &mut R, amount: usize) -> Vec<Self::Item>
    where
        R: Rng + ?Sized,
    {
        let mut reservoir = Vec::from_iter(self.by_ref().take(amount));

        // Continue unless the iterator was exhausted
        //
        // note: this prevents iterators that "restart" from causing problems.
        // If the iterator stops once, then so do we.
        if reservoir.len() == amount {
            for (i, elem) in self.enumerate() {
                let k = rng.random_range(..i + 1 + amount);
                if let Some(slot) = reservoir.get_mut(k) {
                    *slot = elem;
                }
            }
        }
        reservoir
    }

    /// Deprecated: use [`Self::sample_fill`] instead
    #[deprecated(since = "0.10.0", note = "Renamed to `sample_fill`")]
    fn choose_multiple_fill<R>(self, rng: &mut R, buf: &mut [Self::Item]) -> usize
    where
        R: Rng + ?Sized,
    {
        self.sample_fill(rng, buf)
    }

    /// Deprecated: use [`Self::sample`] instead
    #[cfg(feature = "alloc")]
    #[deprecated(since = "0.10.0", note = "Renamed to `sample`")]
    fn choose_multiple<R>(self, rng: &mut R, amount: usize) -> Vec<Self::Item>
    where
        R: Rng + ?Sized,
    {
        self.sample(rng, amount)
    }
}

impl<I> IteratorRandom for I where I: Iterator + Sized {}

#[cfg(test)]
mod test {
    use super::*;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;
    use core::convert::Infallible;

    // Wraps an RNG and counts how many times it is called.
    struct CountingRng<R> {
        inner: R,
        calls: usize,
    }

    impl<R: Rng> crate::TryRng for CountingRng<R> {
        type Error = Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            self.calls += 1;
            Ok(self.inner.next_u32())
        }

        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            self.calls += 1;
            Ok(self.inner.next_u64())
        }

        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
            self.calls += 1;
            self.inner.fill_bytes(dst);
            Ok(())
        }
    }

    #[derive(Clone)]
    struct UnhintedIterator<I: Iterator + Clone> {
        iter: I,
    }
    impl<I: Iterator + Clone> Iterator for UnhintedIterator<I> {
        type Item = I::Item;

        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next()
        }
    }

    // This synthetic iterator could provide an exact size hint, but deliberately
    // omits it to test efficient skipping via `nth` independently of size hints.
    // A real iterator may similarly support fast seeking without knowing its size.
    struct UnhintedIteratorWithFastNth {
        next: u128,
        end: u128,
    }

    impl Iterator for UnhintedIteratorWithFastNth {
        type Item = u128;

        fn next(&mut self) -> Option<Self::Item> {
            self.nth(0)
        }

        fn nth(&mut self, n: usize) -> Option<Self::Item> {
            let value = self.next + n as u128;
            if value >= self.end {
                self.next = self.end;
                None
            } else {
                self.next = value + 1;
                Some(value)
            }
        }
    }

    #[derive(Clone)]
    struct ChunkHintedIterator<I: ExactSizeIterator + Iterator + Clone> {
        iter: I,
        chunk_remaining: usize,
        chunk_size: usize,
        hint_total_size: bool,
    }
    impl<I: ExactSizeIterator + Iterator + Clone> Iterator for ChunkHintedIterator<I> {
        type Item = I::Item;

        fn next(&mut self) -> Option<Self::Item> {
            if self.chunk_remaining == 0 {
                self.chunk_remaining = core::cmp::min(self.chunk_size, self.iter.len());
            }
            self.chunk_remaining = self.chunk_remaining.saturating_sub(1);

            self.iter.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (
                self.chunk_remaining,
                if self.hint_total_size {
                    Some(self.iter.len())
                } else {
                    None
                },
            )
        }
    }

    #[derive(Clone)]
    struct WindowHintedIterator<I: ExactSizeIterator + Iterator + Clone> {
        iter: I,
        window_size: usize,
        hint_total_size: bool,
    }
    impl<I: ExactSizeIterator + Iterator + Clone> Iterator for WindowHintedIterator<I> {
        type Item = I::Item;

        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (
                core::cmp::min(self.iter.len(), self.window_size),
                if self.hint_total_size {
                    Some(self.iter.len())
                } else {
                    None
                },
            )
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Miri is too slow
    fn test_iterator_choose() {
        let r = &mut crate::test::rng(109);
        fn test_iter<R: Rng + ?Sized, Iter: Iterator<Item = usize> + Clone>(r: &mut R, iter: Iter) {
            let mut chosen = [0i32; 9];
            for _ in 0..1000 {
                let picked = iter.clone().choose(r).unwrap();
                chosen[picked] += 1;
            }
            for count in chosen.iter() {
                // Samples should follow Binomial(1000, 1/9)
                // Octave: binopdf(x, 1000, 1/9) gives the prob of *count == x
                // Note: have seen 153, which is unlikely but not impossible.
                assert!(
                    72 < *count && *count < 154,
                    "count not close to 1000/9: {}",
                    count
                );
            }
        }

        test_iter(r, 0..9);
        test_iter(r, [0, 1, 2, 3, 4, 5, 6, 7, 8].iter().cloned());
        #[cfg(feature = "alloc")]
        test_iter(r, (0..9).collect::<Vec<_>>().into_iter());
        test_iter(r, UnhintedIterator { iter: 0..9 });
        test_iter(
            r,
            ChunkHintedIterator {
                iter: 0..9,
                chunk_size: 4,
                chunk_remaining: 4,
                hint_total_size: false,
            },
        );
        test_iter(
            r,
            ChunkHintedIterator {
                iter: 0..9,
                chunk_size: 4,
                chunk_remaining: 4,
                hint_total_size: true,
            },
        );
        test_iter(
            r,
            WindowHintedIterator {
                iter: 0..9,
                window_size: 2,
                hint_total_size: false,
            },
        );
        test_iter(
            r,
            WindowHintedIterator {
                iter: 0..9,
                window_size: 2,
                hint_total_size: true,
            },
        );

        assert_eq!((0..0).choose(r), None);
        assert_eq!(UnhintedIterator { iter: 0..0 }.choose(r), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Miri is too slow
    fn test_iterator_choose_stable() {
        let r = &mut crate::test::rng(109);
        fn test_iter<R: Rng + ?Sized, Iter: Iterator<Item = usize> + Clone>(r: &mut R, iter: Iter) {
            let mut chosen = [0i32; 9];
            for _ in 0..1000 {
                let picked = iter.clone().choose_stable(r).unwrap();
                chosen[picked] += 1;
            }
            for count in chosen.iter() {
                // Samples should follow Binomial(1000, 1/9)
                // Octave: binopdf(x, 1000, 1/9) gives the prob of *count == x
                // Note: have seen 153, which is unlikely but not impossible.
                assert!(
                    72 < *count && *count < 154,
                    "count not close to 1000/9: {}",
                    count
                );
            }
        }

        test_iter(r, 0..9);
        test_iter(r, [0, 1, 2, 3, 4, 5, 6, 7, 8].iter().cloned());
        #[cfg(feature = "alloc")]
        test_iter(r, (0..9).collect::<Vec<_>>().into_iter());
        test_iter(r, UnhintedIterator { iter: 0..9 });
        test_iter(
            r,
            ChunkHintedIterator {
                iter: 0..9,
                chunk_size: 4,
                chunk_remaining: 4,
                hint_total_size: false,
            },
        );
        test_iter(
            r,
            ChunkHintedIterator {
                iter: 0..9,
                chunk_size: 4,
                chunk_remaining: 4,
                hint_total_size: true,
            },
        );
        test_iter(
            r,
            WindowHintedIterator {
                iter: 0..9,
                window_size: 2,
                hint_total_size: false,
            },
        );
        test_iter(
            r,
            WindowHintedIterator {
                iter: 0..9,
                window_size: 2,
                hint_total_size: true,
            },
        );

        assert_eq!((0..0).choose(r), None);
        assert_eq!(UnhintedIterator { iter: 0..0 }.choose(r), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Miri is too slow
    fn test_iterator_choose_stable_stability() {
        fn test_iter(iter: impl Iterator<Item = usize> + Clone) -> [i32; 9] {
            let r = &mut crate::test::rng(109);
            let mut chosen = [0i32; 9];
            for _ in 0..1000 {
                let picked = iter.clone().choose_stable(r).unwrap();
                chosen[picked] += 1;
            }
            chosen
        }

        let reference = test_iter(0..9);
        assert_eq!(
            test_iter([0, 1, 2, 3, 4, 5, 6, 7, 8].iter().cloned()),
            reference
        );

        #[cfg(feature = "alloc")]
        assert_eq!(test_iter((0..9).collect::<Vec<_>>().into_iter()), reference);
        assert_eq!(test_iter(UnhintedIterator { iter: 0..9 }), reference);
        assert_eq!(
            test_iter(ChunkHintedIterator {
                iter: 0..9,
                chunk_size: 4,
                chunk_remaining: 4,
                hint_total_size: false,
            }),
            reference
        );
        assert_eq!(
            test_iter(ChunkHintedIterator {
                iter: 0..9,
                chunk_size: 4,
                chunk_remaining: 4,
                hint_total_size: true,
            }),
            reference
        );
        assert_eq!(
            test_iter(WindowHintedIterator {
                iter: 0..9,
                window_size: 2,
                hint_total_size: false,
            }),
            reference
        );
        assert_eq!(
            test_iter(WindowHintedIterator {
                iter: 0..9,
                window_size: 2,
                hint_total_size: true,
            }),
            reference
        );
    }

    #[test]
    fn test_iterator_choose_stable_skip_boundaries() {
        let mut zero = crate::test::const_rng(0);
        assert_eq!(
            UnhintedIterator { iter: 0..3 }.choose_stable(&mut zero),
            Some(2)
        );

        let mut max = crate::test::const_rng(u64::MAX);
        assert_eq!(
            UnhintedIterator { iter: 0..3 }.choose_stable(&mut max),
            Some(0)
        );
    }

    #[test]
    fn test_nth_u128_above_usize_max() {
        let index = usize::MAX as u128 + 5;
        let mut iter = UnhintedIteratorWithFastNth {
            next: 0,
            end: index + 1,
        };

        assert_eq!(nth_u128(&mut iter, index), Some(index));
    }

    #[test]
    fn test_iterator_choose_stable_rng_efficiency() {
        let mut rng = CountingRng {
            inner: crate::test::rng(123),
            calls: 0,
        };
        let result = UnhintedIterator { iter: 0..1_000_000 }.choose_stable(&mut rng);

        assert!(result.is_some());
        assert!(rng.calls < 100, "used {} RNG calls", rng.calls);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn test_sample_iter() {
        let min_val = 1;
        let max_val = 100;

        let mut r = crate::test::rng(401);
        let vals = (min_val..max_val).collect::<Vec<i32>>();
        let small_sample = vals.iter().sample(&mut r, 5);
        let large_sample = vals.iter().sample(&mut r, vals.len() + 5);

        assert_eq!(small_sample.len(), 5);
        assert_eq!(large_sample.len(), vals.len());
        // no randomization happens when amount >= len
        assert_eq!(large_sample, vals.iter().collect::<Vec<_>>());

        assert!(
            small_sample
                .iter()
                .all(|e| { **e >= min_val && **e <= max_val })
        );
    }

    #[test]
    fn value_stability_choose() {
        fn choose<I: Iterator<Item = u32>>(iter: I) -> Option<u32> {
            let mut rng = crate::test::rng(411);
            iter.choose(&mut rng)
        }

        assert_eq!(choose([].iter().cloned()), None);
        assert_eq!(choose(0..100), Some(33));
        assert_eq!(choose(UnhintedIterator { iter: 0..100 }), Some(77));
        assert_eq!(
            choose(ChunkHintedIterator {
                iter: 0..100,
                chunk_size: 32,
                chunk_remaining: 32,
                hint_total_size: false,
            }),
            Some(77)
        );
        assert_eq!(
            choose(ChunkHintedIterator {
                iter: 0..100,
                chunk_size: 32,
                chunk_remaining: 32,
                hint_total_size: true,
            }),
            Some(77)
        );
        assert_eq!(
            choose(WindowHintedIterator {
                iter: 0..100,
                window_size: 32,
                hint_total_size: false,
            }),
            Some(77)
        );
        assert_eq!(
            choose(WindowHintedIterator {
                iter: 0..100,
                window_size: 32,
                hint_total_size: true,
            }),
            Some(77)
        );
    }

    #[test]
    fn value_stability_choose_stable() {
        fn choose<I: Iterator<Item = u32>>(iter: I) -> Option<u32> {
            let mut rng = crate::test::rng(411);
            iter.choose_stable(&mut rng)
        }

        assert_eq!(choose([].iter().cloned()), None);
        assert_eq!(choose(0..100), Some(77));
        assert_eq!(choose(UnhintedIterator { iter: 0..100 }), Some(77));
        assert_eq!(
            choose(ChunkHintedIterator {
                iter: 0..100,
                chunk_size: 32,
                chunk_remaining: 32,
                hint_total_size: false,
            }),
            Some(77)
        );
        assert_eq!(
            choose(ChunkHintedIterator {
                iter: 0..100,
                chunk_size: 32,
                chunk_remaining: 32,
                hint_total_size: true,
            }),
            Some(77)
        );
        assert_eq!(
            choose(WindowHintedIterator {
                iter: 0..100,
                window_size: 32,
                hint_total_size: false,
            }),
            Some(77)
        );
        assert_eq!(
            choose(WindowHintedIterator {
                iter: 0..100,
                window_size: 32,
                hint_total_size: true,
            }),
            Some(77)
        );
    }

    #[test]
    fn value_stability_sample() {
        fn do_test<I: Clone + Iterator<Item = u32>>(iter: I, v: &[u32]) {
            let mut rng = crate::test::rng(412);
            let mut buf = [0u32; 8];
            assert_eq!(iter.clone().sample_fill(&mut rng, &mut buf), v.len());
            assert_eq!(&buf[0..v.len()], v);

            #[cfg(feature = "alloc")]
            {
                let mut rng = crate::test::rng(412);
                assert_eq!(iter.sample(&mut rng, v.len()), v);
            }
        }

        do_test(0..4, &[0, 1, 2, 3]);
        do_test(0..8, &[0, 1, 2, 3, 4, 5, 6, 7]);
        do_test(0..100, &[77, 95, 38, 23, 25, 8, 58, 40]);
    }
}
