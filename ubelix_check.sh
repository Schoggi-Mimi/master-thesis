#!/usr/bin/env bash
# Overview of the CAIM GPU allocation on UBELIX:
#   1. RUNNING  -- jobs currently running under the QOS
#   2. WAITING  -- jobs still queued, with the reason they wait
#   3. SUMMARY  -- how many GPUs of each type are idle, and how many of those
#                  we may actually take (the QOS has a per-type group quota)
#
# Overrides: QOS=... PART=... TYPES="h200 h100 rtx4090" ./tools/ubelix_check.sh

QOS=${QOS:-job_gpu_caim}
PART=${PART:-gpu-invest}
TYPES=${TYPES:-"h200 h100 rtx4090"}

# ---------------------------------------------------------------- running ----
printf '\n=== RUNNING (qos=%s) ===\n' "$QOS"
squeue -h --qos="$QOS" -t RUNNING -S "u,-M" -o '%i|%u|%P|%N|%D|%b|%M|%L' |
awk -F'|' -v me="$USER" '
  BEGIN{ printf "%-18s %-10s %-11s %-10s %-16s %10s %10s\n",
                "JOBID","USER","PARTITION","NODE","GPUS","RUNTIME","LEFT" }
  {
    gpu=$6; sub(/^gres\/gpu:?/,"",gpu); if(gpu=="N/A"||gpu=="") gpu="-"
    if($5>1) gpu=gpu " x" $5 "nodes"
    printf "%-18s %-10s %-11s %-10s %-16s %10s %10s%s\n",
           $1,$2,$3,$4,gpu,$7,$8,($2==me?"  <- you":"")
    jobs++
    n=split($6,a,":"); if(n>=3){ g[a[2]] += a[3]*$5; tot += a[3]*$5 }
  }
  END{
    if(!jobs){ print "  (none)"; exit }
    s=""; for(t in g) s=s sprintf(" %s=%d",t,g[t])
    printf "  -> %d job(s), %d GPU(s):%s\n", jobs, tot, s
  }'

# ---------------------------------------------------------------- waiting ----
printf '\n=== WAITING (qos=%s) ===\n' "$QOS"
squeue -h --qos="$QOS" -t PENDING -S "-Q" -o '%i|%u|%P|%D|%b|%r|%V|%Q' |
awk -F'|' -v me="$USER" '
  function waited(ts,  s){ s=ts; gsub(/[-T:]/," ",s); return systime()-mktime(s) }
  function dhm(sec,  d,h,m){
    if(sec<0) sec=0
    d=int(sec/86400); h=int((sec%86400)/3600); m=int((sec%3600)/60)
    return d>0 ? sprintf("%dd%02dh",d,h) : sprintf("%d:%02dh",h,m) }
  BEGIN{ printf "%-20s %-10s %-11s %-16s %-20s %8s %10s\n",
                "JOBID","USER","PARTITION","GPUS","REASON","WAIT","PRIORITY" }
  {
    gpu=$5; sub(/^gres\/gpu:?/,"",gpu); if(gpu=="N/A"||gpu=="") gpu="-"
    if($4>1) gpu=gpu " x" $4 "nodes"
    printf "%-20s %-10s %-11s %-16s %-20s %8s %10s%s\n",
           $1,$2,$3,gpu,$6,dhm(waited($7)),$8,($2==me?"  <- you":"")
    jobs++
    n=split($5,a,":"); if(n>=3) g[a[2]] += a[3]*$4
  }
  END{
    if(!jobs){ print "  (none)"; exit }
    s=""; for(t in g) s=s sprintf(" %s=%d",t,g[t])
    printf "  -> %d pending entry/entries (array ranges count as one), GPUs requested:%s\n", jobs, s
  }'

# ---------------------------------------------------------------- summary ----
# QOS group quota and its live usage, e.g. "gres/gpu:h100=8(8)" -> limit 8, used 8.
qos_tres=$(scontrol show assoc_mgr qos="$QOS" flags=qos 2>/dev/null |
           awk '/^[[:space:]]*GrpTRES=/{print; exit}')

printf '\n=== SUMMARY (partition=%s) ===\n' "$PART"
sinfo -h -p "$PART" -N -O "nodehost:20,statecompact:16,gres:70,gresused:70" |
awk -v qos="$qos_tres" -v order="$TYPES" '
  function parse(str, out,  n,i,e,p,c){          # "gpu:h100:8(S:0-1),..." -> out[type]=count
    gsub(/\([^)]*\)/,"",str)
    n=split(str,e,",")
    for(i=1;i<=n;i++){
      if(e[i] !~ /^gpu:/) continue
      p=split(e[i],c,":")
      if(p>=3) out[c[2]] += c[3]; else out["untyped"] += c[2]
    }
  }
  BEGIN{
    # QOS per-type limits / current usage
    n=split(qos,q,",")
    for(i=1;i<=n;i++){
      if(q[i] !~ /gres\/gpu(:|=)/) continue
      split(q[i],kv,"=")
      k=kv[1]; sub(/.*gres\/gpu/,"",k); sub(/^:/,"",k)
      if(k=="") k="__all"
      lim=kv[2]; sub(/\(.*/,"",lim)
      uv=kv[2]; sub(/.*\(/,"",uv); sub(/\)/,"",uv)
      qlim[k]=lim; quse[k]=uv
    }
  }
  {
    node=$1; state=$2; sub(/[*~#%$@+-]+$/,"",state)
    up = (state ~ /^(idle|mix|mixed|alloc|allocated|comp|completing|plnd|planned)$/)
    delete ntot; delete nuse
    parse($3,ntot); parse($4,nuse)
    for(t in ntot){
      seen[t]=1
      if(!up){ off[t] += ntot[t]; offn[t] = offn[t] " " node "(" state ")"; continue }
      T[t] += ntot[t]; U[t] += nuse[t]
      f = ntot[t] - nuse[t]
      if(f>0) freen[t] = freen[t] sprintf(" %s(%d)", node, f)
    }
  }
  END{
    # display order: TYPES first, then whatever else the partition has
    n=split(order,o," "); k=0
    for(i=1;i<=n;i++) if(o[i] in seen || o[i] in qlim){ list[++k]=o[i]; done[o[i]]=1 }
    for(t in seen) if(!(t in done)){ list[++k]=t; done[t]=1 }

    # the QOS also caps the total number of GPUs, whatever their type
    allleft = ("__all" in qlim && qlim["__all"] != "N") ? qlim["__all"]-quse["__all"] : -1
    if(allleft < 0 && allleft != -1) allleft = 0

    printf "%-14s %6s %6s %6s %8s %8s %7s\n",
           "GPU TYPE","TOTAL","USED","IDLE","QUOTA","QOS_USE","FOR_YOU"
    for(i=1;i<=k;i++){
      t=list[i]; idle=T[t]-U[t]
      if(t in qlim && qlim[t] != "N"){ left=qlim[t]-quse[t]; if(left<0) left=0
                                       avail=(left<idle?left:idle); ql=qlim[t]; qu=quse[t] }
      else { avail=idle; ql="-"; qu="-" }
      if(allleft != -1 && avail > allleft) avail = allleft
      printf "%-14s %6d %6d %6d %8s %8s %7d%s\n",
             substr(t,1,14), T[t], U[t], idle, ql, qu, avail, (avail>0?"  <<":"")
      if(idle>0) detail = detail sprintf("  idle %-14s:%s\n", substr(t,1,14), freen[t])
      if(off[t]>0) offline = offline sprintf("  down %-14s: %d GPU(s) on%s\n", substr(t,1,14), off[t], offn[t])
    }
    if("__all" in qlim && qlim["__all"] != "N")
      printf "%-14s %6s %6s %6s %8s %8s %7s\n",
             "(any gpu)","-","-","-",qlim["__all"],quse["__all"],qlim["__all"]-quse["__all"]
    printf "\n"
    if(detail)  printf "%s", detail
    if(offline) printf "%s", offline
    printf "  TOTAL/USED/IDLE count only usable nodes; QUOTA/QOS_USE are the QOS group\n"
    printf "  limits, so FOR_YOU = min(IDLE, QUOTA-QOS_USE) is what a new job can get.\n"
  }'
