# Model Release Checklist

All items are intentionally unchecked in this staging repository. The release
owner must record evidence and approval for every applicable item.

## Ownership, license, and data

- [ ] Name the accountable release owner and approvers.
- [ ] Owner reviews and approves the final release license.
- [ ] Verify the Mistral base-model license, acceptable-use terms, attribution,
      and redistribution requirements against the exact downloaded revision.
- [ ] Complete provenance records for every training, validation, and test
      dataset.
- [ ] Owner confirms collection, processing, modification, and redistribution
      rights for all training data.
- [ ] Complete privacy, consent, retention, deletion, and sensitive-data review.
- [ ] Complete copyright, biometric-data, and jurisdiction-specific legal review.
- [ ] Confirm that no dataset, raw audio, transcript, annotation export, or
      personal information is included in the model repository.

## Artifact integrity

- [ ] Export weights with the approved exporter and package version.
- [ ] Confirm the weights contain the backbone, ASR head, and `vad_lm_head`.
- [ ] Confirm there are no optimizer, scheduler, gradient, trainer-state, or
      intermediate-checkpoint artifacts.
- [ ] Validate `params.json`, tokenizer, processor, and generation metadata
      against the exported weights.
- [ ] Replace the documentation-only `config.example.json` only if the serving
      package requires a validated `config.json`.
- [ ] Run the model-definition unit tests from
      `voxtral-realtime/integrations/transformers` and reconstruct the
      canonical checkpoint with `load_mtp_checkpoint`.
- [ ] Verify the output-label order is exactly `idle`, `noidle`, `speaking`,
      `turn_end`, `backchannel`, `uncertain`.
- [ ] Verify the frame duration is 80 ms end-to-end.
- [ ] Generate and archive file hashes and artifact sizes.
- [ ] Scan all files for secrets, credentials, usernames, hostnames, private
      paths, internal URLs, logs, and embedded training examples.
- [ ] Run `python verify_model_repo.py --allow-weights` after adding approved
      release weights.

## Evaluation and documentation

- [ ] Replace every evaluation placeholder in `README.md` with reviewed results
      or an explicit reason for omission.
- [ ] Document datasets, splits, rights, preprocessing, decoding, thresholds,
      frame alignment, package versions, hardware, and random seeds.
- [ ] Report ASR and turn-taking metrics without cherry-picking.
- [ ] Evaluate Chinese, English, mixed-language, noise, overlap, accents,
      dialects, far-field audio, and relevant subgroup slices.
- [ ] Measure streaming latency and throughput under a reproducible serving
      configuration.
- [ ] Review intended use, out-of-scope use, limitations, privacy, bias, and
      failure modes with domain owners.
- [ ] Confirm examples do not expose private or copyrighted data.

## Serving and release

- [ ] Pin a compatible `voxtral-realtime` package version and validate a clean
      installation.
- [ ] Run offline, streaming, long-audio, malformed-input, and concurrency smoke
      tests.
- [ ] Verify application policy does not act irrevocably on a single turn frame.
- [ ] Define monitoring, rollback, abuse reporting, security response, and model
      deprecation processes.
- [ ] Review repository visibility, access controls, and destination namespace.
- [ ] Obtain final owner sign-off immediately before public upload.
- [ ] Record the released git revision and remote model-repository revision.
