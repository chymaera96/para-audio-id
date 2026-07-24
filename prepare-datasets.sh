#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${1:-/gpfs/scratch/acw723/audio-degradation-data-sources}"
OUTPUT_ROOT="${2:-/gpfs/scratch/acw723/audio-degradation-data/degradation_24k}"
WORKERS="${3:-${SLURM_CPUS_PER_TASK:-8}}"
ARCHIVE_ROOT="${WORK_ROOT}/archives"
RAW_ROOT="${WORK_ROOT}/raw"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command is unavailable: $1" >&2
    exit 1
  fi
}

download() {
  local url="$1"
  local destination="$2"
  mkdir -p "$(dirname -- "${destination}")"
  if [[ -s "${destination}" ]]; then
    echo "Using existing archive: ${destination}"
    return
  fi
  echo "Downloading ${url}"
  wget --continue --tries=8 --timeout=60 --output-document="${destination}" "${url}"
}

extract_zip() {
  local archive="$1"
  local destination="$2"
  mkdir -p "${destination}"
  unzip -n -q "${archive}" -d "${destination}"
}

require_command python
require_command wget
require_command unzip
require_command md5sum
mkdir -p "${ARCHIVE_ROOT}" "${RAW_ROOT}" "${OUTPUT_ROOT}"

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  # TUT Acoustic Scenes 2016: original 44.1 kHz development and evaluation audio.
  TUT_DEV_MD5=(
    e39546e65f2e72517b6335aaf0c8323d d36cf3253e2c041f68e937a3fe804807
    0393a9620ab882b1c26d884eccdcffdd fb3e4e0cd7ea82120ec07031dee558ce
    a19cf600b33c8f88f6ad607bafd74057 591aad3219d1155342572cc1f6af5680
    9e6c1897789e6bce13ac69c6caedb7ab c4718354f48fcc9dfc7305f6cd8325c8
  )
  for part in {1..8}; do
    name="TUT-acoustic-scenes-2016-development.audio.${part}.zip"
    archive="${ARCHIVE_ROOT}/tut/${name}"
    download "https://zenodo.org/records/45739/files/${name}?download=1" "${archive}"
    echo "${TUT_DEV_MD5[$((part - 1))]}  ${archive}" | md5sum --check -
    extract_zip "${archive}" "${RAW_ROOT}/tut"
  done

  TUT_EVAL_MD5=(
    7c6c2e54b8a9c4c37a803b81446d16fe 7930f1dc26707ab3ba9526073af87333
    17187d633d6402aee4b481122a1b28f0
  )
  for part in {1..3}; do
    name="TUT-acoustic-scenes-2016-evaluation.audio.${part}.zip"
    archive="${ARCHIVE_ROOT}/tut/${name}"
    download "https://zenodo.org/records/165995/files/${name}?download=1" "${archive}"
    echo "${TUT_EVAL_MD5[$((part - 1))]}  ${archive}" | md5sum --check -
    extract_zip "${archive}" "${RAW_ROOT}/tut"
  done

  # MIT Survey: 270 original room impulse responses.
  archive="${ARCHIVE_ROOT}/mit/Audio.zip"
  download "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip" "${archive}"
  extract_zip "${archive}" "${RAW_ROOT}/mit"

  # OpenAIR: download original WAV files only. The compiler later retains mono/stereo IRs.
  if ! find "${RAW_ROOT}/openair" -type f -iname '*.wav' -print -quit 2>/dev/null | grep -q .; then
    mkdir -p "${RAW_ROOT}/openair"
    wget --mirror --no-parent --no-host-directories --continue \
      --accept='*.wav' --reject='*example*,*Examples*' \
      --directory-prefix="${RAW_ROOT}/openair" \
      https://webfiles.york.ac.uk/OPENAIR/IRs/
  fi

  # Aachen AIR v1.4: official 48 kHz MAT/WAV distribution.
  archive="${ARCHIVE_ROOT}/aachen/air_database_release_1_4.zip"
  download \
    "https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/air_database_release_1_4.zip" \
    "${archive}"
  extract_zip "${archive}" "${RAW_ROOT}/aachen"

  # Surrey microphone IRs: original normalized/raw 48 kHz, 24/32-bit distribution.
  archive="${ARCHIVE_ROOT}/microphone/Microphone_Impulse_Responses.zip"
  download \
    "https://zenodo.org/records/4633508/files/Microphone_Impulse_Responses.zip?download=1" \
    "${archive}"
  echo "c66c4f46be850aa25022145431b93caa  ${archive}" | md5sum --check -
  extract_zip "${archive}" "${RAW_ROOT}/microphone"
else
  echo "SKIP_DOWNLOAD=1: using sources already extracted under ${RAW_ROOT}"
fi

python "${SCRIPT_DIR}/scripts/prepare_degradation_audio.py" \
  "${RAW_ROOT}" \
  "${OUTPUT_ROOT}" \
  --sample-rate 24000 \
  --workers "${WORKERS}"

echo
echo "Prepared original-source 24 kHz degradation data: ${OUTPUT_ROOT}"
echo "Downloaded archives and extracted originals: ${WORK_ROOT}"
echo "Manifest: ${OUTPUT_ROOT}/manifest.jsonl"
echo "Summary: ${OUTPUT_ROOT}/summary.json"
echo "Bad files: ${OUTPUT_ROOT}/bad_files.jsonl"
