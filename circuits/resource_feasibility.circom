pragma circom 2.2.0;

include "poseidon.circom";
include "bitify.circom";
include "comparators.circom";

template ChainPoseidon(n) {
    signal input domain;
    signal input fields[n];
    signal output out;

    component hashers[n];
    signal state[n + 1];
    state[0] <== domain;

    for (var i = 0; i < n; i++) {
        hashers[i] = Poseidon(2);
        hashers[i].inputs[0] <== state[i];
        hashers[i].inputs[1] <== fields[i];
        state[i + 1] <== hashers[i].out;
    }

    out <== state[n];
}

function PairSrc(p, VN) {
    // Upper-triangular pair order: (0,1), (0,2), ..., (VN-2,VN-1).
    var src = 0;
    var pairIdx = 0;
    for (var u = 0; u < VN; u++) {
        for (var v = u + 1; v < VN; v++) {
            if (pairIdx == p) {
                src = u;
            }
            pairIdx++;
        }
    }
    return src;
}

function PairDst(p, VN) {
    var dst = 0;
    var pairIdx = 0;
    for (var u = 0; u < VN; u++) {
        for (var v = u + 1; v < VN; v++) {
            if (pairIdx == p) {
                dst = v;
            }
            pairIdx++;
        }
    }
    return dst;
}

// range check
template RequestChecks(VN, SN_BITS) {
    signal input reqId;
    signal input vNodePresent[VN];
    signal input vCpu[VN];
    signal input vMem[VN];
    signal input vArch[VN];
    signal input vLinkPresent[VN][VN];
    signal input vBw[VN][VN];
    signal input endpointEnabled;
    signal input endpointIdx;

    component reqIdBits = Num2Bits(16);
    reqIdBits.in <== reqId;

    component vCpuBits[VN];
    component vMemBits[VN];
    component vArchBits[VN];
    component vBwBits[VN][VN];
    for (var v = 0; v < VN; v++) {
        vNodePresent[v] * (vNodePresent[v] - 1) === 0;
        vCpuBits[v] = Num2Bits(16);
        vCpuBits[v].in <== vCpu[v];
        vMemBits[v] = Num2Bits(16);
        vMemBits[v].in <== vMem[v];
        vArchBits[v] = Num2Bits(8);
        vArchBits[v].in <== vArch[v];
        vCpu[v] * (1 - vNodePresent[v]) === 0;
        vMem[v] * (1 - vNodePresent[v]) === 0;
        vArch[v] * (1 - vNodePresent[v]) === 0;
        for (var u = 0; u < VN; u++) {
            vLinkPresent[v][u] * (vLinkPresent[v][u] - 1) === 0;
            vLinkPresent[v][u] * (1 - vNodePresent[v]) === 0;
            vLinkPresent[v][u] * (1 - vNodePresent[u]) === 0;
            vBwBits[v][u] = Num2Bits(16);
            vBwBits[v][u].in <== vBw[v][u];
            vBw[v][u] * (1 - vLinkPresent[v][u]) === 0;
        }
        vLinkPresent[v][v] === 0;
        vBw[v][v] === 0;
    }
    for (var v = 0; v < VN; v++) {
        for (var u = v + 1; u < VN; u++) {
            vLinkPresent[v][u] === vLinkPresent[u][v];
            vBw[v][u] === vBw[u][v];
        }
    }

    endpointEnabled * (endpointEnabled - 1) === 0;
    component endpointBits = Num2Bits(SN_BITS);
    endpointBits.in <== endpointIdx;
}

template SubstrateStateChecks(SN) {
    signal input sNodePresent[SN];
    signal input sCpuBefore[SN];
    signal input sCpuAfter[SN];
    signal input sMemBefore[SN];
    signal input sMemAfter[SN];
    signal input sArch[SN];
    signal input sAdj[SN][SN];
    signal input sBwBefore[SN][SN];
    signal input sBwAfter[SN][SN];

    component sCpuBeforeBits[SN];
    component sCpuAfterBits[SN];
    component sMemBeforeBits[SN];
    component sMemAfterBits[SN];
    component sArchBits[SN];
    component sBwBeforeBits[SN][SN];
    component sBwAfterBits[SN][SN];
    for (var s = 0; s < SN; s++) {
        sNodePresent[s] * (sNodePresent[s] - 1) === 0;

        sCpuBeforeBits[s] = Num2Bits(16);
        sCpuBeforeBits[s].in <== sCpuBefore[s];
        sCpuAfterBits[s] = Num2Bits(16);
        sCpuAfterBits[s].in <== sCpuAfter[s];
        sMemBeforeBits[s] = Num2Bits(16);
        sMemBeforeBits[s].in <== sMemBefore[s];
        sMemAfterBits[s] = Num2Bits(16);
        sMemAfterBits[s].in <== sMemAfter[s];
        sArchBits[s] = Num2Bits(8);
        sArchBits[s].in <== sArch[s];

        sCpuBefore[s] * (1 - sNodePresent[s]) === 0;
        sCpuAfter[s] * (1 - sNodePresent[s]) === 0;
        sMemBefore[s] * (1 - sNodePresent[s]) === 0;
        sMemAfter[s] * (1 - sNodePresent[s]) === 0;
        sArch[s] * (1 - sNodePresent[s]) === 0;

        for (var t = 0; t < SN; t++) {
            sAdj[s][t] * (sAdj[s][t] - 1) === 0;
            sAdj[s][t] * (1 - sNodePresent[s]) === 0;
            sAdj[s][t] * (1 - sNodePresent[t]) === 0;
            sBwBeforeBits[s][t] = Num2Bits(16);
            sBwBeforeBits[s][t].in <== sBwBefore[s][t];
            sBwAfterBits[s][t] = Num2Bits(16);
            sBwAfterBits[s][t].in <== sBwAfter[s][t];

            sBwBefore[s][t] * (1 - sNodePresent[s]) === 0;
            sBwBefore[s][t] * (1 - sNodePresent[t]) === 0;
            sBwAfter[s][t] * (1 - sNodePresent[s]) === 0;
            sBwAfter[s][t] * (1 - sNodePresent[t]) === 0;
            sBwBefore[s][t] * (1 - sAdj[s][t]) === 0;
            sBwAfter[s][t] * (1 - sAdj[s][t]) === 0;
        }
        sAdj[s][s] === 0;
        sBwBefore[s][s] === 0;
        sBwAfter[s][s] === 0;
    }
    for (var s = 0; s < SN; s++) {
        for (var t = s + 1; t < SN; t++) {
            sAdj[s][t] === sAdj[t][s];
            sBwBefore[s][t] === sBwBefore[t][s];
            sBwAfter[s][t] === sBwAfter[t][s];
        }
    }
}

template NodeMappingChecks(VN, SN, SN_BITS) {
    signal input vNodePresent[VN];
    signal input vArch[VN];
    signal input sNodePresent[SN];
    signal input sArch[SN];
    signal input endpointEnabled;
    signal input endpointIdx;
    signal input nodeMap[VN];
    signal input nodeSel[VN][SN];

    component nodeMapBits[VN];
    component archRequiredIsZero[VN];
    signal archRequired[VN];
    signal selectedArchTerm[VN][SN];
    signal selectedArch[VN];
    for (var v = 0; v < VN; v++) {
        nodeMapBits[v] = Num2Bits(SN_BITS);
        nodeMapBits[v].in <== nodeMap[v];
    }

    for (var v = 0; v < VN; v++) {
        var nodeSelSum = 0;
        var nodeMapSum = 0;
        var selectedArchSum = 0;
        for (var s = 0; s < SN; s++) {
            nodeSel[v][s] * (nodeSel[v][s] - 1) === 0;
            nodeSel[v][s] * (1 - sNodePresent[s]) === 0;
            nodeSelSum += nodeSel[v][s];
            nodeMapSum += s * nodeSel[v][s];
            selectedArchTerm[v][s] <== sArch[s] * nodeSel[v][s];
            selectedArchSum += selectedArchTerm[v][s];
        }
        nodeSelSum === vNodePresent[v];
        nodeMap[v] === nodeMapSum;
        selectedArch[v] <== selectedArchSum;
        archRequiredIsZero[v] = IsZero();
        archRequiredIsZero[v].in <== vArch[v];
        archRequired[v] <== 1 - archRequiredIsZero[v].out;
        (selectedArch[v] - vArch[v]) * archRequired[v] === 0;
    }

    endpointEnabled * (1 - vNodePresent[VN - 1]) === 0;
    (nodeMap[VN - 1] - endpointIdx) * endpointEnabled === 0;
}

template NodeResourceReservationChecks(VN, SN) {
    signal input vCpu[VN];
    signal input vMem[VN];
    signal input sCpuBefore[SN];
    signal input sCpuAfter[SN];
    signal input sMemBefore[SN];
    signal input sMemAfter[SN];
    signal input nodeSel[VN][SN];

    signal selectedCpuTerm[VN][SN];
    signal selectedMemTerm[VN][SN];
    signal usedCpu[SN];
    signal usedMem[SN];
    component cpuCapacityLe[SN];
    component memCapacityLe[SN];

    for (var s = 0; s < SN; s++) {
        var cpuUseSum = 0;
        var memUseSum = 0;
        for (var v = 0; v < VN; v++) {
            selectedCpuTerm[v][s] <== vCpu[v] * nodeSel[v][s];
            cpuUseSum += selectedCpuTerm[v][s];
            selectedMemTerm[v][s] <== vMem[v] * nodeSel[v][s];
            memUseSum += selectedMemTerm[v][s];
        }
        usedCpu[s] <== cpuUseSum;
        usedMem[s] <== memUseSum;
        sCpuAfter[s] + usedCpu[s] === sCpuBefore[s];
        sMemAfter[s] + usedMem[s] === sMemBefore[s];

        cpuCapacityLe[s] = LessEqThan(20);
        cpuCapacityLe[s].in[0] <== usedCpu[s];
        cpuCapacityLe[s].in[1] <== sCpuBefore[s];
        cpuCapacityLe[s].out === 1;

        memCapacityLe[s] = LessEqThan(20);
        memCapacityLe[s].in[0] <== usedMem[s];
        memCapacityLe[s].in[1] <== sMemBefore[s];
        memCapacityLe[s].out === 1;
    }
}

template PathWitnessChecks(VN, SN, SN_BITS, PAIR_COUNT, HOP_COUNT) {
    signal input vLinkPresent[VN][VN];
    signal input sNodePresent[SN];
    signal input nodeMap[VN];
    signal input pathNodes[PAIR_COUNT][HOP_COUNT + 1];
    signal input pathNodeSel[PAIR_COUNT][HOP_COUNT + 1][SN];
    // Index h selects a path with exactly h substrate hops, including h=0.
    signal input pathLenSel[PAIR_COUNT][HOP_COUNT + 1];

    component pathNodeBits[PAIR_COUNT][HOP_COUNT + 1];
    signal edgeActive[PAIR_COUNT];
    signal pathLenSum[PAIR_COUNT];
    signal pathPosActive[PAIR_COUNT][HOP_COUNT + 1];
    signal pathEndTerm[PAIR_COUNT][HOP_COUNT + 1];
    signal pathEndNode[PAIR_COUNT];

    for (var p = 0; p < PAIR_COUNT; p++) {
        var srcVnode = PairSrc(p, VN);
        var dstVnode = PairDst(p, VN);

        edgeActive[p] <== vLinkPresent[srcVnode][dstVnode];

        var lenSum = 0;
        for (var h = 0; h < HOP_COUNT + 1; h++) {
            pathLenSel[p][h] * (pathLenSel[p][h] - 1) === 0;
            lenSum += pathLenSel[p][h];
        }
        pathLenSum[p] <== lenSum;
        pathLenSum[p] === edgeActive[p];

        for (var pos = 0; pos < HOP_COUNT + 1; pos++) {
            pathNodeBits[p][pos] = Num2Bits(SN_BITS);
            pathNodeBits[p][pos].in <== pathNodes[p][pos];

            var posActiveSum = 0;
            for (var lenIdx = 0; lenIdx < HOP_COUNT + 1; lenIdx++) {
                if (pos <= lenIdx) {
                    posActiveSum += pathLenSel[p][lenIdx];
                }
            }
            pathPosActive[p][pos] <== posActiveSum;

            var pathNodeSelSum = 0;
            var pathNodeIdxSum = 0;
            for (var s = 0; s < SN; s++) {
                pathNodeSel[p][pos][s] * (pathNodeSel[p][pos][s] - 1) === 0;
                pathNodeSel[p][pos][s] * (1 - sNodePresent[s]) === 0;
                pathNodeSelSum += pathNodeSel[p][pos][s];
                pathNodeIdxSum += s * pathNodeSel[p][pos][s];
            }
            pathNodeSelSum === pathPosActive[p][pos];
            pathNodes[p][pos] === pathNodeIdxSum;
        }

        pathNodes[p][0] === nodeMap[srcVnode] * edgeActive[p];

        var endSum = 0;
        for (var h = 0; h < HOP_COUNT + 1; h++) {
            pathEndTerm[p][h] <== pathLenSel[p][h] * pathNodes[p][h];
            endSum += pathEndTerm[p][h];
        }
        pathEndNode[p] <== endSum;
        pathEndNode[p] === nodeMap[dstVnode] * edgeActive[p];
    }
}

template BandwidthReservationChecks(VN, SN, PAIR_COUNT, HOP_COUNT) {
    signal input vBw[VN][VN];
    signal input sAdj[SN][SN];
    signal input sBwBefore[SN][SN];
    signal input sBwAfter[SN][SN];
    signal input pathNodeSel[PAIR_COUNT][HOP_COUNT + 1][SN];

    signal hopForward[PAIR_COUNT][HOP_COUNT][SN][SN];
    signal hopReverse[PAIR_COUNT][HOP_COUNT][SN][SN];
    signal hopDemandTerm[PAIR_COUNT][HOP_COUNT][SN][SN];
    signal usedBw[SN][SN];
    component bwCapacityLe[SN][SN];

    for (var s = 0; s < SN; s++) {
        for (var t = 0; t < SN; t++) {
            var bwUseSum = 0;
            for (var p = 0; p < PAIR_COUNT; p++) {
                var srcVnode = PairSrc(p, VN);
                var dstVnode = PairDst(p, VN);

                for (var h = 0; h < HOP_COUNT; h++) {
                    hopForward[p][h][s][t] <== pathNodeSel[p][h][s] * pathNodeSel[p][h + 1][t];
                    hopReverse[p][h][s][t] <== pathNodeSel[p][h][t] * pathNodeSel[p][h + 1][s];
                    (hopForward[p][h][s][t] + hopReverse[p][h][s][t]) * (1 - sAdj[s][t]) === 0;
                    hopDemandTerm[p][h][s][t] <== (
                        hopForward[p][h][s][t] + hopReverse[p][h][s][t]
                    ) * vBw[srcVnode][dstVnode];
                    bwUseSum += hopDemandTerm[p][h][s][t];
                }
            }
            usedBw[s][t] <== bwUseSum;

            if (s == t) {
                usedBw[s][t] === 0;
            }

            sBwAfter[s][t] + usedBw[s][t] === sBwBefore[s][t];

            bwCapacityLe[s][t] = LessEqThan(20);
            bwCapacityLe[s][t].in[0] <== usedBw[s][t];
            bwCapacityLe[s][t].in[1] <== sBwBefore[s][t];
            bwCapacityLe[s][t].out === 1;
        }
    }
}

template CommitmentChecks(VN, SN, PAIR_COUNT, HOP_COUNT) {
    signal input reqId;
    signal input vNodePresent[VN];
    signal input vCpu[VN];
    signal input vMem[VN];
    signal input vArch[VN];
    signal input vLinkPresent[VN][VN];
    signal input vBw[VN][VN];
    signal input endpointEnabled;
    signal input endpointIdx;

    signal input C_r;
    signal input C_S;
    signal input C_M;
    signal input C_S_next;

    signal input sNodePresent[SN];
    signal input sCpuBefore[SN];
    signal input sCpuAfter[SN];
    signal input sMemBefore[SN];
    signal input sMemAfter[SN];
    signal input sArch[SN];
    signal input sAdj[SN][SN];
    signal input sBwBefore[SN][SN];
    signal input sBwAfter[SN][SN];

    signal input nodeMap[VN];
    signal input pathNodes[PAIR_COUNT][HOP_COUNT + 1];
    signal input pathLenSel[PAIR_COUNT][HOP_COUNT + 1];

    signal input rho_S;
    signal input rho_M;
    signal input rho_S_next;

    component requestCommit = ChainPoseidon(3 + 4 * VN + 2 * VN * VN);
    requestCommit.domain <== 1001;
    var reqIdx = 0;
    requestCommit.fields[reqIdx] <== reqId;
    reqIdx = reqIdx + 1;
    for (var v = 0; v < VN; v++) {
        requestCommit.fields[reqIdx] <== vNodePresent[v];
        reqIdx = reqIdx + 1;
    }
    for (var v = 0; v < VN; v++) {
        requestCommit.fields[reqIdx] <== vCpu[v];
        reqIdx = reqIdx + 1;
    }
    for (var v = 0; v < VN; v++) {
        requestCommit.fields[reqIdx] <== vMem[v];
        reqIdx = reqIdx + 1;
    }
    for (var v = 0; v < VN; v++) {
        requestCommit.fields[reqIdx] <== vArch[v];
        reqIdx = reqIdx + 1;
    }
    for (var v = 0; v < VN; v++) {
        for (var u = 0; u < VN; u++) {
            requestCommit.fields[reqIdx] <== vLinkPresent[v][u];
            reqIdx = reqIdx + 1;
        }
    }
    for (var v = 0; v < VN; v++) {
        for (var u = 0; u < VN; u++) {
            requestCommit.fields[reqIdx] <== vBw[v][u];
            reqIdx = reqIdx + 1;
        }
    }
    requestCommit.fields[reqIdx] <== endpointEnabled;
    reqIdx = reqIdx + 1;
    requestCommit.fields[reqIdx] <== endpointIdx;
    requestCommit.out === C_r;

    component substrateCommit = ChainPoseidon(4 * SN + 2 * SN * SN + 1);
    substrateCommit.domain <== 1002;
    var subIdx = 0;
    for (var s = 0; s < SN; s++) {
        substrateCommit.fields[subIdx] <== sNodePresent[s];
        subIdx = subIdx + 1;
    }
    for (var s = 0; s < SN; s++) {
        substrateCommit.fields[subIdx] <== sCpuBefore[s];
        subIdx = subIdx + 1;
    }
    for (var s = 0; s < SN; s++) {
        substrateCommit.fields[subIdx] <== sMemBefore[s];
        subIdx = subIdx + 1;
    }
    for (var s = 0; s < SN; s++) {
        substrateCommit.fields[subIdx] <== sArch[s];
        subIdx = subIdx + 1;
    }
    for (var s = 0; s < SN; s++) {
        for (var t = 0; t < SN; t++) {
            substrateCommit.fields[subIdx] <== sAdj[s][t];
            subIdx = subIdx + 1;
        }
    }
    for (var s = 0; s < SN; s++) {
        for (var t = 0; t < SN; t++) {
            substrateCommit.fields[subIdx] <== sBwBefore[s][t];
            subIdx = subIdx + 1;
        }
    }
    substrateCommit.fields[subIdx] <== rho_S;
    substrateCommit.out === C_S;

    component embeddingCommit = ChainPoseidon(VN + PAIR_COUNT * (HOP_COUNT + 1) + PAIR_COUNT * (HOP_COUNT + 1) + 1);
    embeddingCommit.domain <== 1003;
    var embIdx = 0;
    for (var v = 0; v < VN; v++) {
        embeddingCommit.fields[embIdx] <== nodeMap[v];
        embIdx = embIdx + 1;
    }
    for (var p = 0; p < PAIR_COUNT; p++) {
        for (var h = 0; h < HOP_COUNT + 1; h++) {
            embeddingCommit.fields[embIdx] <== pathLenSel[p][h];
            embIdx = embIdx + 1;
        }
    }
    for (var p = 0; p < PAIR_COUNT; p++) {
        for (var pos = 0; pos < HOP_COUNT + 1; pos++) {
            embeddingCommit.fields[embIdx] <== pathNodes[p][pos];
            embIdx = embIdx + 1;
        }
    }
    embeddingCommit.fields[embIdx] <== rho_M;
    embeddingCommit.out === C_M;

    component substrateNextCommit = ChainPoseidon(4 * SN + 2 * SN * SN + 1);
    substrateNextCommit.domain <== 1002;
    var nextIdx = 0;
    for (var s = 0; s < SN; s++) {
        substrateNextCommit.fields[nextIdx] <== sNodePresent[s];
        nextIdx = nextIdx + 1;
    }
    for (var s = 0; s < SN; s++) {
        substrateNextCommit.fields[nextIdx] <== sCpuAfter[s];
        nextIdx = nextIdx + 1;
    }
    for (var s = 0; s < SN; s++) {
        substrateNextCommit.fields[nextIdx] <== sMemAfter[s];
        nextIdx = nextIdx + 1;
    }
    for (var s = 0; s < SN; s++) {
        substrateNextCommit.fields[nextIdx] <== sArch[s];
        nextIdx = nextIdx + 1;
    }
    for (var s = 0; s < SN; s++) {
        for (var t = 0; t < SN; t++) {
            substrateNextCommit.fields[nextIdx] <== sAdj[s][t];
            nextIdx = nextIdx + 1;
        }
    }
    for (var s = 0; s < SN; s++) {
        for (var t = 0; t < SN; t++) {
            substrateNextCommit.fields[nextIdx] <== sBwAfter[s][t];
            nextIdx = nextIdx + 1;
        }
    }
    substrateNextCommit.fields[nextIdx] <== rho_S_next;
    substrateNextCommit.out === C_S_next;
}

template ResourceFeasibility(VN, SN, SN_BITS, PAIR_COUNT, HOP_COUNT) {
    signal input reqId;
    signal input vNodePresent[VN];
    signal input vCpu[VN];
    signal input vMem[VN];
    signal input vArch[VN];
    signal input vLinkPresent[VN][VN];
    signal input vBw[VN][VN];
    signal input endpointEnabled;
    signal input endpointIdx;

    signal input C_r;
    signal input C_S;
    signal input C_M;
    signal input C_S_next;

    signal input sNodePresent[SN];
    signal input sCpuBefore[SN];
    signal input sCpuAfter[SN];
    signal input sMemBefore[SN];
    signal input sMemAfter[SN];
    signal input sArch[SN];
    signal input sAdj[SN][SN];
    signal input sBwBefore[SN][SN];
    signal input sBwAfter[SN][SN];

    signal input nodeMap[VN];
    signal input nodeSel[VN][SN];
    signal input pathNodes[PAIR_COUNT][HOP_COUNT + 1];
    signal input pathNodeSel[PAIR_COUNT][HOP_COUNT + 1][SN];
    signal input pathLenSel[PAIR_COUNT][HOP_COUNT + 1];

    signal input rho_S;
    signal input rho_M;
    signal input rho_S_next;

    component requestChecks = RequestChecks(VN, SN_BITS);
    requestChecks.reqId <== reqId;
    requestChecks.endpointEnabled <== endpointEnabled;
    requestChecks.endpointIdx <== endpointIdx;
    for (var v = 0; v < VN; v++) {
        requestChecks.vNodePresent[v] <== vNodePresent[v];
        requestChecks.vCpu[v] <== vCpu[v];
        requestChecks.vMem[v] <== vMem[v];
        requestChecks.vArch[v] <== vArch[v];
        for (var u = 0; u < VN; u++) {
            requestChecks.vLinkPresent[v][u] <== vLinkPresent[v][u];
            requestChecks.vBw[v][u] <== vBw[v][u];
        }
    }

    component substrateChecks = SubstrateStateChecks(SN);
    for (var s = 0; s < SN; s++) {
        substrateChecks.sNodePresent[s] <== sNodePresent[s];
        substrateChecks.sCpuBefore[s] <== sCpuBefore[s];
        substrateChecks.sCpuAfter[s] <== sCpuAfter[s];
        substrateChecks.sMemBefore[s] <== sMemBefore[s];
        substrateChecks.sMemAfter[s] <== sMemAfter[s];
        substrateChecks.sArch[s] <== sArch[s];
        for (var t = 0; t < SN; t++) {
            substrateChecks.sAdj[s][t] <== sAdj[s][t];
            substrateChecks.sBwBefore[s][t] <== sBwBefore[s][t];
            substrateChecks.sBwAfter[s][t] <== sBwAfter[s][t];
        }
    }

    component nodeMappingChecks = NodeMappingChecks(VN, SN, SN_BITS);
    nodeMappingChecks.endpointEnabled <== endpointEnabled;
    nodeMappingChecks.endpointIdx <== endpointIdx;
    for (var s = 0; s < SN; s++) {
        nodeMappingChecks.sNodePresent[s] <== sNodePresent[s];
        nodeMappingChecks.sArch[s] <== sArch[s];
    }
    for (var v = 0; v < VN; v++) {
        nodeMappingChecks.vNodePresent[v] <== vNodePresent[v];
        nodeMappingChecks.vArch[v] <== vArch[v];
        nodeMappingChecks.nodeMap[v] <== nodeMap[v];
        for (var s = 0; s < SN; s++) {
            nodeMappingChecks.nodeSel[v][s] <== nodeSel[v][s];
        }
    }

    component resourceChecks = NodeResourceReservationChecks(VN, SN);
    for (var s = 0; s < SN; s++) {
        resourceChecks.sCpuBefore[s] <== sCpuBefore[s];
        resourceChecks.sCpuAfter[s] <== sCpuAfter[s];
        resourceChecks.sMemBefore[s] <== sMemBefore[s];
        resourceChecks.sMemAfter[s] <== sMemAfter[s];
    }
    for (var v = 0; v < VN; v++) {
        resourceChecks.vCpu[v] <== vCpu[v];
        resourceChecks.vMem[v] <== vMem[v];
        for (var s = 0; s < SN; s++) {
            resourceChecks.nodeSel[v][s] <== nodeSel[v][s];
        }
    }

    component pathChecks = PathWitnessChecks(VN, SN, SN_BITS, PAIR_COUNT, HOP_COUNT);
    for (var v = 0; v < VN; v++) {
        pathChecks.nodeMap[v] <== nodeMap[v];
        for (var u = 0; u < VN; u++) {
            pathChecks.vLinkPresent[v][u] <== vLinkPresent[v][u];
        }
    }
    for (var s = 0; s < SN; s++) {
        pathChecks.sNodePresent[s] <== sNodePresent[s];
    }
    for (var p = 0; p < PAIR_COUNT; p++) {
        for (var h = 0; h < HOP_COUNT + 1; h++) {
            pathChecks.pathLenSel[p][h] <== pathLenSel[p][h];
        }
        for (var pos = 0; pos < HOP_COUNT + 1; pos++) {
            pathChecks.pathNodes[p][pos] <== pathNodes[p][pos];
            for (var s = 0; s < SN; s++) {
                pathChecks.pathNodeSel[p][pos][s] <== pathNodeSel[p][pos][s];
            }
        }
    }

    component bwChecks = BandwidthReservationChecks(VN, SN, PAIR_COUNT, HOP_COUNT);
    for (var v = 0; v < VN; v++) {
        for (var u = 0; u < VN; u++) {
            bwChecks.vBw[v][u] <== vBw[v][u];
        }
    }
    for (var s = 0; s < SN; s++) {
        for (var t = 0; t < SN; t++) {
            bwChecks.sAdj[s][t] <== sAdj[s][t];
            bwChecks.sBwBefore[s][t] <== sBwBefore[s][t];
            bwChecks.sBwAfter[s][t] <== sBwAfter[s][t];
        }
    }
    for (var p = 0; p < PAIR_COUNT; p++) {
        for (var pos = 0; pos < HOP_COUNT + 1; pos++) {
            for (var s = 0; s < SN; s++) {
                bwChecks.pathNodeSel[p][pos][s] <== pathNodeSel[p][pos][s];
            }
        }
    }

    component commitmentChecks = CommitmentChecks(VN, SN, PAIR_COUNT, HOP_COUNT);
    commitmentChecks.reqId <== reqId;
    commitmentChecks.endpointEnabled <== endpointEnabled;
    commitmentChecks.endpointIdx <== endpointIdx;
    commitmentChecks.C_r <== C_r;
    commitmentChecks.C_S <== C_S;
    commitmentChecks.C_M <== C_M;
    commitmentChecks.C_S_next <== C_S_next;
    commitmentChecks.rho_S <== rho_S;
    commitmentChecks.rho_M <== rho_M;
    commitmentChecks.rho_S_next <== rho_S_next;
    for (var v = 0; v < VN; v++) {
        commitmentChecks.vNodePresent[v] <== vNodePresent[v];
        commitmentChecks.vCpu[v] <== vCpu[v];
        commitmentChecks.vMem[v] <== vMem[v];
        commitmentChecks.vArch[v] <== vArch[v];
        commitmentChecks.nodeMap[v] <== nodeMap[v];
        for (var u = 0; u < VN; u++) {
            commitmentChecks.vLinkPresent[v][u] <== vLinkPresent[v][u];
            commitmentChecks.vBw[v][u] <== vBw[v][u];
        }
    }
    for (var s = 0; s < SN; s++) {
        commitmentChecks.sNodePresent[s] <== sNodePresent[s];
        commitmentChecks.sCpuBefore[s] <== sCpuBefore[s];
        commitmentChecks.sCpuAfter[s] <== sCpuAfter[s];
        commitmentChecks.sMemBefore[s] <== sMemBefore[s];
        commitmentChecks.sMemAfter[s] <== sMemAfter[s];
        commitmentChecks.sArch[s] <== sArch[s];
        for (var t = 0; t < SN; t++) {
            commitmentChecks.sAdj[s][t] <== sAdj[s][t];
            commitmentChecks.sBwBefore[s][t] <== sBwBefore[s][t];
            commitmentChecks.sBwAfter[s][t] <== sBwAfter[s][t];
        }
    }
    for (var p = 0; p < PAIR_COUNT; p++) {
        for (var h = 0; h < HOP_COUNT + 1; h++) {
            commitmentChecks.pathLenSel[p][h] <== pathLenSel[p][h];
        }
        for (var pos = 0; pos < HOP_COUNT + 1; pos++) {
            commitmentChecks.pathNodes[p][pos] <== pathNodes[p][pos];
        }
    }
}

component main { public [
    reqId,
    vNodePresent,
    vCpu,
    vMem,
    vArch,
    vLinkPresent,
    vBw,
    endpointEnabled,
    endpointIdx,
    C_r,
    C_S,
    C_M,
    C_S_next
] } = ResourceFeasibility(3, 6, 3, 3, 3);
