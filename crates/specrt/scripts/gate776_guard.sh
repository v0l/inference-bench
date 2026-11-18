gpu_busy_mib=${GPU_BUSY_MIB:-4000}
gpu_app_mib=${GPU_APP_MIB:-512}

gpu_inventory() {
    if [ -n "${GATE_GPUS_FILE:-}" ]; then
        cat "$GATE_GPUS_FILE"
    else
        nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits
    fi
}

gpu_apps() {
    if [ -n "${GATE_APPS_FILE:-}" ]; then
        cat "$GATE_APPS_FILE"
    else
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
    fi
}

gpu_busy_report() {
    local gpus apps
    gpus=$(gpu_inventory)
    apps=$(gpu_apps)
    awk -F', *' -v busy="$gpu_busy_mib" -v appmib="$gpu_app_mib" '
        FILENAME == ARGV[1] {
            index_of[$2] = $1
            used[$1] = $3
            next
        }
        {
            mem = $NF
            name = $3
            for (i = 4; i < NF; i++) name = name ", " $i
            if (mem + 0 >= appmib) {
                gpu = ($1 in index_of) ? index_of[$1] : $1
                printf "another process holds GPU %s (pid %s, %s, %s MiB): free both GPUs first\n", gpu, $2, name, mem
                found = 1
            }
        }
        END {
            for (g in used) if (used[g] + 0 > busy) {
                printf "GPU %s has %s MiB in use: free both GPUs first\n", g, used[g]
                found = 1
            }
            exit found ? 1 : 0
        }
    ' <(printf '%s\n' "$gpus") <(printf '%s\n' "$apps")
}

require_idle_gpus() {
    local report status=0
    report=$(gpu_busy_report) || status=$?
    if [ "$status" -ne 0 ]; then
        printf '%s\n' "$report" >&2
        return 1
    fi
}
