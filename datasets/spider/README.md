# Spider SQLite databases — clean copy

`Data/Spider/database/` (the 166 Spider SQLite databases used by every SQL-track
script and notebook) is gitignored, since it's ~870MB uncompressed — too large
to track directly. This file is the pointer to where the corrected copy lives.

**Drive link**: https://drive.google.com/file/d/1A6adY6ubQWyPeY3_d9EXuvZ3-q4RiprS/view?usp=share_link

## What's "corrected" about it

`flight_2`'s `Flights.SourceAirport`/`DestAirport` columns were stored with a
leading space that `Airports.AirportCode` and every gold-query literal lack,
so any `WHERE`/`JOIN` comparing them silently returned 0/empty instead of
erroring — gold itself scored wrong on 42/1034 held-out SQL dev questions
(4.1%). Scanned all 166 databases; `flight_2` is the only one where this
reaches a column the held-out dev set actually filters/joins on. Fixed via
`TRIM()` on the 6 affected columns (`airports.City/AirportName/Country/
CountryAbbrev`, `flights.SourceAirport/DestAirport`). Full root-cause writeup
in the PR that landed the eval-side fix (`src/eval/harness.py` +
`notebooks/phase18_eval_ablation_res.ipynb`).

## Usage

Same drop-in path every notebook already expects:

```
!cp /content/drive/MyDrive/codegen/checkpoints/spider_database.zip /content/Codegen/
!unzip -q /content/Codegen/spider_database.zip -d /content/Codegen/Data/Spider/
```

If you replace `spider_database.zip` on Drive with this fixed copy (same
filename, same path), no notebook changes are needed — every existing cell
keeps working. `notebooks/phase18_eval_ablation_res.ipynb`'s Section 3b also
re-applies the same `TRIM()` fix defensively after unzipping; it's a no-op on
already-clean data, so it's safe to leave in regardless of which zip you use.
