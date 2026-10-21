# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reclaim the ray and vLLM processes belonging to one launcher slot.
#
# vLLM's server does not die with its parent: nothing sets PR_SET_PDEATHSIG, and the engine cores
# it spawns are its own children, so a ray job killed with SIGKILL leaves the whole tree adopted by
# init and still holding device memory. These processes are matched by the VIME_LAUNCHER_RUN_ID
# marker that launcher.sh puts in the ray runtime env, and signalled by process group so the engine
# cores go with the root.
#
# Sourced by launcher.sh; usable on its own to clear a slot by hand:
#   VIME_LAUNCHER_RUN_ID=<script>:<ray temp dir> VIME_RAY_TEMP_DIR=<dir> VIME_RAY_PORT=<port> \
#       bash -c 'source scripts/torchtitan/common/reclaim.sh; reclaim_slot'

ray_port_busy() {
  timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/${VIME_RAY_PORT}" 2>/dev/null
}

launcher_vllm_pgids() {
  # Discovery is environ-based, not name-based. The VIME_LAUNCHER_RUN_ID marker is inherited by
  # every descendant of the ray job (vllm serve root, DP coordinator, engine cores, API servers,
  # workers) and survives setproctitle rewrites. Name matching ($3 == "vllm" && "vllm serve " in
  # args) can only see a LIVE root process: once the root dies the orphaned DP children are
  # retitled to "VLLM::EngineCore_DP*", so both comm and args stop matching.
  #
  # grep flags: -z treats environ as NUL-separated lines, -F -x requires one entry to equal the
  # marker exactly, so a RUN_ID that is a prefix of another slot's (ray/bi2 vs ray/bi20) cannot
  # cross-match. Unreadable environ (another user's processes) is skipped by the -r guard.
  local proc pid pgid marker self_pgid
  marker="VIME_LAUNCHER_RUN_ID=${VIME_LAUNCHER_RUN_ID}"
  self_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"

  for proc in /proc/[0-9]*; do
    pid="${proc#/proc/}"
    [[ -r "${proc}/environ" ]] || continue
    grep -azFxq -- "${marker}" "${proc}/environ" 2>/dev/null || continue
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
    [[ -n "${pgid}" ]] || continue
    # Never signal the launcher's own process group.
    [[ "${pgid}" != "${self_pgid}" ]] || continue
    printf '%s\n' "${pgid}"
  done | sort -u
}

reclaim_slot() {
  local pids deadline pgid

  # Without the ownership marker there is nothing to match on, so cleanup is a no-op rather than
  # falling back to a broad process match.
  [[ -n "${VIME_LAUNCHER_RUN_ID:-}" ]] || return 0
  [[ -n "${VIME_RAY_TEMP_DIR:-}" ]] || return 0

  while read -r pgid; do
    [[ -n "${pgid}" ]] || continue
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  done < <(launcher_vllm_pgids)

  sleep 2
  while read -r pgid; do
    [[ -n "${pgid}" ]] || continue
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  done < <(launcher_vllm_pgids)

  pids=$(ps -eo pid=,cmd= | awk -v temp_dir="${VIME_RAY_TEMP_DIR}" \
    'index($0, temp_dir) { print $1 }')
  [[ -z "${pids}" ]] || kill -9 ${pids} 2>/dev/null || true

  deadline=$((SECONDS + 60))
  while ray_port_busy; do
    (( SECONDS < deadline )) || {
      echo "Ray port ${VIME_RAY_PORT} is still busy after cleanup" >&2
      return 1
    }
    sleep 2
  done
}
