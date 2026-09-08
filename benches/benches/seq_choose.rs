// Copyright 2018-2023 Developers of the Rand project.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::prelude::*;
use rand_pcg::Pcg32;

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench
);
criterion_main!(benches);

pub fn bench(c: &mut Criterion) {
    c.bench_function("seq_slice_choose_1_of_100", |b| {
        let mut rng: Pcg32 = rand::make_rng();
        let mut buf = [0i32; 100];
        rng.fill(&mut buf);
        let x = black_box(&mut buf);

        b.iter(|| x.choose(&mut rng).unwrap());
    });

    let lens = [(1, 1000), (950, 1000), (10, 100), (90, 100)];
    for (amount, len) in lens {
        let name = format!("seq_slice_sample_{amount}_of_{len}");
        c.bench_function(name.as_str(), |b| {
            let mut rng: Pcg32 = rand::make_rng();
            let mut buf = [0i32; 1000];
            rng.fill(&mut buf);
            let x = black_box(&buf[..len]);

            let mut results_buf = [0i32; 950];
            let y = black_box(&mut results_buf[..amount]);
            let amount = black_box(amount);

            b.iter(|| {
                // Collect full result to prevent unwanted shortcuts getting
                // first element (in case sample_indices returns an iterator).
                for (slot, sample) in y.iter_mut().zip(x.sample(&mut rng, amount)) {
                    *slot = *sample;
                }
                y[amount - 1]
            })
        });
    }

    let lens = [(1, 1000), (950, 1000), (10, 100), (90, 100)];
    for (amount, len) in lens {
        let name = format!("seq_slice_sample_weighted_{amount}_of_{len}");
        c.bench_function(name.as_str(), |b| {
            let mut rng: Pcg32 = rand::make_rng();
            let mut buf = [0i32; 1000];
            rng.fill(&mut buf);
            let x = black_box(&buf[..len]);

            let mut results_buf = [0i32; 950];
            let y = black_box(&mut results_buf[..amount]);
            let amount = black_box(amount);

            b.iter(|| {
                // Collect full result to prevent unwanted shortcuts getting
                // first element (in case sample_indices returns an iterator).
                let samples_iter = x.sample_weighted(&mut rng, amount, |_| 1.0).unwrap();
                for (slot, sample) in y.iter_mut().zip(samples_iter) {
                    *slot = *sample;
                }
                y[amount - 1]
            })
        });
    }

    c.bench_function("seq_iter_sample_10_of_100", |b| {
        let mut rng: Pcg32 = rand::make_rng();
        let mut buf = [0i32; 100];
        rng.fill(&mut buf);
        let x = black_box(&buf);
        b.iter(|| x.iter().cloned().sample(&mut rng, 10))
    });

    c.bench_function("seq_iter_sample_fill_10_of_100", |b| {
        let mut rng: Pcg32 = rand::make_rng();
        let mut buf = [0i32; 100];
        rng.fill(&mut buf);
        let x = black_box(&buf);
        let mut buf = [0; 10];
        b.iter(|| x.iter().cloned().sample_fill(&mut rng, &mut buf))
    });

    bench_rng::<chacha20::ChaCha20Rng>(c, "ChaCha20");
    bench_rng::<rand_pcg::Pcg32>(c, "Pcg32");
    bench_rng::<rand_pcg::Pcg64>(c, "Pcg64");

    bench_fast_nth(c);
    bench_unhinted(c);
}

fn bench_fast_nth(c: &mut Criterion) {
    let mut group = c.benchmark_group("choose_fast_nth");

    // This models a large iterator which can seek efficiently but cannot report
    // its remaining length. Keep the largest input tractable for the linear-time
    // implementations so that all four algorithms can be compared directly.
    for length in [1_000, 1_000_000, 10_000_000] {
        group.bench_with_input(BenchmarkId::new("choose_old", length), &length, |b, &length| {
            let mut rng = Pcg32::seed_from_u64(123);
            b.iter(|| choose_old(UnhintedIteratorWithFastNth::new(length), &mut rng))
        });

        group.bench_with_input(BenchmarkId::new("choose_stable_old", length), &length, |b, &length| {
            let mut rng = Pcg32::seed_from_u64(123);
            b.iter(|| choose_stable_old(UnhintedIteratorWithFastNth::new(length), &mut rng))
        });

        group.bench_with_input(BenchmarkId::new("choose_stable_new", length), &length, |b, &length| {
            let mut rng = Pcg32::seed_from_u64(123);
            b.iter(|| UnhintedIteratorWithFastNth::new(length).choose_stable(&mut rng))
        });

        group.bench_with_input(BenchmarkId::new("choose_new", length), &length, |b, &length| {
            let mut rng = Pcg32::seed_from_u64(123);
            b.iter(|| UnhintedIteratorWithFastNth::new(length).choose(&mut rng))
        });
    }
}

fn bench_unhinted(c: &mut Criterion) {
    let mut group = c.benchmark_group("choose_unhinted");

    // Unlike `UnhintedIteratorWithFastNth`, this iterator uses the default
    // linear-time `nth`. This checks the candidate's general streaming case.
    for length in [1_000, 1_000_000, 10_000_000] {
        group.bench_with_input(BenchmarkId::new("choose_old", length), &length, |b, &length| {
            let mut rng = Pcg32::seed_from_u64(123);
            b.iter(|| choose_old(UnhintedIterator { iter: 0..length }, &mut rng))
        });

        group.bench_with_input(BenchmarkId::new("choose_stable_old", length), &length, |b, &length| {
            let mut rng = Pcg32::seed_from_u64(123);
            b.iter(|| choose_stable_old(UnhintedIterator { iter: 0..length }, &mut rng))
        });

        group.bench_with_input(BenchmarkId::new("choose_stable_new", length), &length, |b, &length| {
            let mut rng = Pcg32::seed_from_u64(123);
            b.iter(|| UnhintedIterator { iter: 0..length }.choose_stable(&mut rng))
        });

        group.bench_with_input(BenchmarkId::new("choose_new", length), &length, |b, &length| {
            let mut rng = Pcg32::seed_from_u64(123);
            b.iter(|| UnhintedIterator { iter: 0..length }.choose(&mut rng))
        });
    }
}

fn bench_rng<R: Rng + SeedableRng>(c: &mut Criterion, rng_name: &'static str) {
    for length in [1, 2, 3, 10, 100, 1000].map(black_box) {
        let name = format!("choose_size-hinted_from_{length}_{rng_name}");
        c.bench_function(name.as_str(), |b| {
            let mut rng = R::seed_from_u64(123);
            b.iter(|| choose_size_hinted(length, &mut rng))
        });

        let name = format!("choose_stable_from_{length}_{rng_name}");
        c.bench_function(name.as_str(), |b| {
            let mut rng = R::seed_from_u64(123);
            b.iter(|| choose_stable(length, &mut rng))
        });

        let name = format!("choose_unhinted_from_{length}_{rng_name}");
        c.bench_function(name.as_str(), |b| {
            let mut rng = R::seed_from_u64(123);
            b.iter(|| choose_unhinted(length, &mut rng))
        });

        let name = format!("choose_windowed_from_{length}_{rng_name}");
        c.bench_function(name.as_str(), |b| {
            let mut rng = R::seed_from_u64(123);
            b.iter(|| choose_windowed(length, 7, &mut rng))
        });
    }
}

fn choose_size_hinted<R: Rng>(max: usize, rng: &mut R) -> Option<usize> {
    let iterator = 0..max;
    iterator.choose(rng)
}

fn choose_stable<R: Rng>(max: usize, rng: &mut R) -> Option<usize> {
    let iterator = 0..max;
    iterator.choose_stable(rng)
}

fn choose_unhinted<R: Rng>(max: usize, rng: &mut R) -> Option<usize> {
    let iterator = UnhintedIterator { iter: (0..max) };
    iterator.choose(rng)
}

fn choose_windowed<R: Rng>(max: usize, window_size: usize, rng: &mut R) -> Option<usize> {
    let iterator = WindowHintedIterator {
        iter: (0..max),
        window_size,
    };
    iterator.choose(rng)
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

#[derive(Clone)]
struct WindowHintedIterator<I: ExactSizeIterator + Iterator + Clone> {
    iter: I,
    window_size: usize,
}
impl<I: ExactSizeIterator + Iterator + Clone> Iterator for WindowHintedIterator<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (core::cmp::min(self.iter.len(), self.window_size), None)
    }
}

#[derive(Clone)]
struct UnhintedIteratorWithFastNth {
    next: usize,
    end: usize,
}

impl UnhintedIteratorWithFastNth {
    fn new(end: usize) -> Self {
        Self { next: 0, end }
    }
}

impl Iterator for UnhintedIteratorWithFastNth {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let value = self.next.saturating_add(n);
        if value >= self.end {
            self.next = self.end;
            None
        } else {
            self.next = value + 1;
            Some(value)
        }
    }
}

// The implementation of `choose` before it delegated unknown-size iterators to
// skip-based reservoir sampling.
fn choose_old<I, R>(mut iter: I, rng: &mut R) -> Option<I::Item>
where
    I: Iterator,
    R: Rng + ?Sized,
{
    let (mut lower, mut upper) = iter.size_hint();
    let mut result = None;

    if upper == Some(lower) {
        return match lower {
            0 => None,
            1 => iter.next(),
            _ => iter.nth(rng.random_range(..lower)),
        };
    }

    let mut coin_flipper = OldCoinFlipper::new(rng);
    let mut consumed = 0;

    loop {
        if lower > 1 {
            let index = coin_flipper.rng.random_range(..lower + consumed);
            let skip = if index < lower {
                result = iter.nth(index);
                lower - (index + 1)
            } else {
                lower
            };
            if upper == Some(lower) {
                return result;
            }
            consumed += lower;
            if skip > 0 {
                iter.nth(skip - 1);
            }
        } else {
            let element = iter.next();
            if element.is_none() {
                return result;
            }
            consumed += 1;
            if coin_flipper.random_ratio_one_over(consumed) {
                result = element;
            }
        }

        let hint = iter.size_hint();
        lower = hint.0;
        upper = hint.1;
    }
}

// The implementation of `choose_stable` before skip-based reservoir sampling.
// Keeping it here gives Criterion a stable comparison target after the library
// implementation changes.
fn choose_stable_old<I, R>(mut iter: I, rng: &mut R) -> Option<I::Item>
where
    I: Iterator,
    R: Rng + ?Sized,
{
    let mut consumed = 0;
    let mut result = None;
    let mut coin_flipper = OldCoinFlipper::new(rng);

    loop {
        let mut next = 0;
        let (lower, _) = iter.size_hint();
        if lower >= 2 {
            let highest_selected = (0..lower)
                .filter(|ix| coin_flipper.random_ratio_one_over(consumed + ix + 1))
                .last();

            consumed += lower;
            next = lower;

            if let Some(ix) = highest_selected {
                result = iter.nth(ix);
                next -= ix + 1;
            }
        }

        let elem = iter.nth(next);
        if elem.is_none() {
            return result;
        }

        if coin_flipper.random_ratio_one_over(consumed + 1) {
            result = elem;
        }
        consumed += 1;
    }
}

struct OldCoinFlipper<R> {
    rng: R,
    chunk: u32,
    chunk_remaining: u32,
}

impl<R: Rng> OldCoinFlipper<R> {
    fn new(rng: R) -> Self {
        Self {
            rng,
            chunk: 0,
            chunk_remaining: 0,
        }
    }

    #[inline]
    fn random_ratio_one_over(&mut self, denominator: usize) -> bool {
        let flips = (usize::BITS - 1 - denominator.leading_zeros()).min(32);
        if self.flip_heads(flips) {
            self.random_ratio(1 << flips, denominator)
        } else {
            false
        }
    }

    #[inline]
    fn random_ratio(&mut self, mut numerator: usize, denominator: usize) -> bool {
        while numerator < denominator {
            let flips = numerator
                .leading_zeros()
                .saturating_sub(denominator.leading_zeros() + 1)
                .clamp(1, 32);

            if self.flip_heads(flips) {
                numerator = numerator.saturating_mul(2_usize.pow(flips));
            } else if flips == 1 {
                let next = numerator.wrapping_add(numerator).wrapping_sub(denominator);
                if next == 0 || next > numerator {
                    return false;
                }
                numerator = next;
            } else {
                return false;
            }
        }
        true
    }

    fn flip_heads(&mut self, mut flips: u32) -> bool {
        loop {
            let zeros = self.chunk.leading_zeros();
            if zeros < flips {
                self.chunk = self.chunk.wrapping_shl(zeros + 1);
                self.chunk_remaining = self.chunk_remaining.saturating_sub(zeros + 1);
                return false;
            } else if let Some(remaining) = self.chunk_remaining.checked_sub(flips) {
                self.chunk_remaining = remaining;
                self.chunk <<= flips;
                return true;
            } else {
                flips -= self.chunk_remaining;
                self.chunk = self.rng.next_u32();
                self.chunk_remaining = 32;
            }
        }
    }
}
