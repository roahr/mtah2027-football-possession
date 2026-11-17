#!/bin/bash
# Download the public tracking data of OTHER matches used to train and validate the supervised
# models (see solution/external.py). Nothing here is the competition match.
#
#   IDSSE  DFL / Sportec Solutions open data, 7 matches, CC BY 4.0. Bassek et al., Scientific Data
#          12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y
#          (mirror used: huggingface.co/datasets/pysport/idsse-data)
#   Metrica Sports sample data, 3 matches: github.com/metrica-sports/sample-data
#   SkillCorner open data, 20 A-League 2024/25 matches, broadcast tracking, MIT licence:
#          github.com/SkillCorner/opendata (pinned to commit 4340d27; tracking via Git LFS)
#
# Files land in data/external/ (git-ignored) and are checked against solution/external_sha256.txt.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/../data/external/idsse" "$HERE/../data/external/metrica" "$HERE/../data/external/skillcorner"
cd "$HERE/../data/external"

fetch() {  # url destination
  [ -s "$2" ] && return 0
  curl -sfL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

HF=https://huggingface.co/datasets/pysport/idsse-data/resolve/main
for m in J03WMX:DFL-COM-000001 J03WN1:DFL-COM-000001 J03WOH:DFL-COM-000002 J03WOY:DFL-COM-000002 \
         J03WPY:DFL-COM-000002 J03WQQ:DFL-COM-000002 J03WR9:DFL-COM-000002; do
  id=${m%%:*}; c=${m##*:}
  (fetch "$HF/DFL_02_01_matchinformation_${c}_DFL-MAT-${id}.xml" "idsse/DFL_02_01_matchinformation_${c}_DFL-MAT-${id}.xml"
   fetch "$HF/DFL_04_03_positions_raw_observed_${c}_DFL-MAT-${id}.xml" "idsse/DFL_04_03_positions_raw_observed_${c}_DFL-MAT-${id}.xml") &
done

GH=https://raw.githubusercontent.com/metrica-sports/sample-data/master/data
for f in Sample_Game_1/Sample_Game_1_RawEventsData.csv Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv \
         Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv Sample_Game_2/Sample_Game_2_RawEventsData.csv \
         Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv \
         Sample_Game_3/Sample_Game_3_events.json Sample_Game_3/Sample_Game_3_metadata.xml Sample_Game_3/Sample_Game_3_tracking.txt; do
  fetch "$GH/$f" "metrica/$(basename "$f")" &
done
wait

SC_REV=4340d274572876239c154c90bc507a9b3250a656
SC_RAW=https://raw.githubusercontent.com/SkillCorner/opendata/$SC_REV/data/matches
SC_LFS=https://media.githubusercontent.com/media/SkillCorner/opendata/$SC_REV/data/matches
for m in 1874553 1886347 1899585 1925299 1927964 1953632 1959846 1986691 1996435 1996436 \
         2006229 2006363 2007448 2007721 2010085 2011166 2013725 2015213 2016236 2017461; do
  (fetch "$SC_RAW/$m/${m}_match.json" "skillcorner/${m}_match.json"
   fetch "$SC_RAW/$m/${m}_phases_of_play.csv" "skillcorner/${m}_phases_of_play.csv"
   fetch "$SC_LFS/$m/${m}_tracking_extrapolated.jsonl" "skillcorner/${m}_tracking_extrapolated.jsonl") &
done
wait
shasum -a 256 -c "$HERE/external_sha256.txt"
