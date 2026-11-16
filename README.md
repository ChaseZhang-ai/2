<div align="center">

# ZhuluAI · 逐鹿

**An autonomous content-operations engine and B-end automation platform for SMB e-commerce.**

[![CI](https://github.com/zixinzenith/zhulu-ai/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#quickstart)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-42%20passing-brightgreen.svg)](tests)
[![Dependencies](https://img.shields.io/badge/runtime%20deps-none-success.svg)](#design-notes)

*Built in 2025, open-sourced in 2026.*

</div>

> Turn a flat product photo into a model shot, a marketing clip and a talking-head
> video; answer customer questions in eight languages from the merchant's own
> knowledge base; and run the merchant's internal approvals, ticket routing and
> data-sync as automated workflows — all on one RTX 4090.

> **A note on this repo's history.** The engine was built and operated during 2025.
> The commit history here is the **2026 open-source cleanup** (Sept–Nov 2026): the
> code was re-committed module by module as it was prepared for release, so the
> commit dates reflect the release work, not the original 2025 build timeline.

This repository is the engine behind **ZhuluAI (逐鹿 AI)**, a content-operations
startup I founded in 2025. The company is winding down its managed-service arm,
so I am releasing the core engine: the parts that were genuinely hard, technical
and worth showing. See [docs/comparison.md](docs/comparison.md) for how this
differs from "just another AI video generator" and from a traditional MCN.

---

## Contents

- [What it does](#what-it-does)
- [Results](#results)
- [Why the naive approach does not work](#why-the-naive-approach-does-not-work)
- [Architecture](#architecture)
- [What is interesting here, technically](#what-is-interesting-here-technically)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Benchmarks](#benchmarks)
- [Design notes](#design-notes)
- [Honest limitations](#honest-limitations)
- [License](#license)

---

