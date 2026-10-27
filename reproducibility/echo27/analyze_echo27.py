"""Frozen new-batch reader. No network; raw 9-bit records, no postselection.

Input: {origin: IBM_REAL_RAW9, blocks: [{block:0,settings:[{name,raw9:[...256]}...]}...]}
Importer must separately verify actual circuit order/QPY/options/complete job provenance.
Default self-test uses labeled in-memory arithmetic fixtures, never hardware data.
"""
import sys
sys.dont_write_bytecode=True
import argparse
from pathlib import Path
from fractions import Fraction as F
import json,math

HERE=Path(__file__).resolve().parent
M=json.loads((HERE/"MEASUREMENT_MANIFEST_NOT_AUTHORIZED.json").read_text())
SPEC={s["name"]:s for s in M["settings"]}
ALPHA=.05; J=23; N=1024
R=2*math.sqrt(math.log(2*J/ALPHA)/(2*N))

def score(raw,s):
    mask=sum(1<<(5+q) for q in s["obs"][1])
    return int(s["obs"][2])*(1-2*((int(raw,2)&mask).bit_count()%2))

def analyze(data,fixture=False):
    assert data["origin"]==("ARITHMETIC_FIXTURE_NOT_IBM" if fixture else "IBM_REAL_RAW9")
    blocks=data["blocks"]; assert len(blocks)==4 and {b["block"] for b in blocks}==set(range(4))
    sums={n:0 for n,s in SPEC.items() if s["kind"]=="science"}; perblock=[]; cal=[]
    for block in sorted(blocks,key=lambda b:b["block"]):
        rows=block["settings"]; idx=block["block"]
        assert [r["name"] for r in rows]==M["blocks"][idx]["order"]
        assert {r["name"] for r in rows}==set(SPEC) and len(rows)==27
        local={}
        for row in rows:
            s=SPEC[row["name"]]; raw=row["raw9"]
            assert len(raw)==256 and all(isinstance(x,str) and len(x)==9 and set(x)<=set("01") for x in raw)
            if s["kind"]=="science":
                num=sum(score(x,s) for x in raw); sums[s["name"]]+=num
                local[s["name"]]=F(num,256)
            else:
                matches=sum(int(x,2)==s["pattern"]<<5 for x in raw)
                cal.append(dict(block=idx,name=s["name"],whole9_match_fraction=matches/256))
        perblock.append(dict(block=idx,axis_means={n:float(x) for n,x in local.items()},
            group_nominal_scores={g:float((1+sum(local[f"{g}_W{k}"] for k in range(3)))/4)
                                  for g in ("BELL0","IDLE_T","ECHO_T","P5_T","SEP0")}))
    means={n:F(v,N) for n,v in sums.items()}; assert len(means)==J
    axes={n:dict(mean=float(x),integer_sum=sums[n],n=N,lower=float(x)-R,upper=float(x)+R) for n,x in means.items()}
    forms={g:{f"{g}_W{k}":F(1,4) for k in range(3)} for g in ("BELL0","IDLE_T","ECHO_T","P5_T","SEP0")}
    def lin(weights,constant=F(0)):
        value=constant+sum(w*means[n] for n,w in weights.items()); rad=float(sum(abs(w) for w in weights.values()))*R
        return dict(mean=float(value),exact_fraction=str(value),radius=rad,lower=float(value)-rad,upper=float(value)+rad)
    groups={g:lin(v,F(1,4)) for g,v in forms.items()}
    def combine(coeff):
        out={}
        for g,c in coeff.items():
            for n,w in forms[g].items(): out[n]=out.get(n,F(0))+c*w
        return lin(out,F(1,4)*sum(coeff.values()))
    contrasts={
        "primary_D":combine({"ECHO_T":F(1),"BELL0":F(-1,2),"IDLE_T":F(-1,2)}),
        "waiting_loss_BELL0_minus_IDLE":combine({"BELL0":F(1),"IDLE_T":F(-1)}),
        "ECHO_minus_IDLE":combine({"ECHO_T":F(1),"IDLE_T":F(-1)}),
        "P5_minus_ECHO":combine({"P5_T":F(1),"ECHO_T":F(-1)})}
    d=contrasts["primary_D"]; loss=contrasts["waiting_loss_BELL0_minus_IDLE"]
    verdict="SUPPORT_HALF_RECOVERY" if d["lower"]>0 and loss["lower"]>0 else (
        "REJECT_HALF_RECOVERY_FOR_IMPLEMENTED_PROTOCOL" if d["upper"]<0 and loss["lower"]>0 else "UNRESOLVED")
    return dict(origin=data["origin"],axes=axes,groups=groups,contrasts=contrasts,per_block=perblock,
        calibration_descriptive_only=cal,alpha=ALPHA,J=J,N_per_axis=N,coordinate_radius=R,verdict=verdict,
        primary_algebra_sign="positive" if d["lower"]>0 else "negative" if d["upper"]<0 else "unresolved",
        fractional_interpretation_allowed=loss["lower"]>0,
        estimand="Average conditional expected signed readings at actual setting events; not automatically common-state fidelity.",
        block_independence_or_stationarity_not_assumed=True,entanglement_certification=False,
        phase_fitted=False,postselection=False,pooled_with_old_jobs=False)

def fixture():
    # Known exact means: BELL W=1, IDLE W=0, ECHO/P5 W=.5, SEP W=0, all phase=0.
    blocks=[]
    for b in M["blocks"]:
        rows=[]
        for name in b["order"]:
            s=SPEC[name]
            if s["kind"]=="calibration": raw=[f'{s["pattern"]<<5:09b}']*256
            else:
                plus=256 if name.startswith("BELL0_W") else 192 if name.startswith(("ECHO_T_W","P5_T_W")) else 128
                bit=1<<(5+s["obs"][1][0]); positive=0 if s["obs"][2]==1 else bit
                negative=positive^bit
                raw=[f"{positive:09b}"]*plus+[f"{negative:09b}"]*(256-plus)
            rows.append(dict(name=name,raw9=raw))
        blocks.append(dict(block=b["block"],settings=rows))
    return dict(origin="ARITHMETIC_FIXTURE_NOT_IBM",blocks=blocks)

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path);p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.input:
        assert args.output and args.input.resolve()!=args.output.resolve()
        result=analyze(json.loads(args.input.read_text()))
        with args.output.open("x") as stream: json.dump(result,stream,indent=2);stream.write("\n")
    else:
        data=fixture(); r=analyze(data,fixture=True)
        assert r["groups"]["BELL0"]["exact_fraction"]=="1"
        assert r["groups"]["IDLE_T"]["exact_fraction"]=="1/4"
        assert r["groups"]["ECHO_T"]["exact_fraction"]=="5/8"
        assert r["contrasts"]["primary_D"]["exact_fraction"]=="0"
        assert abs(r["contrasts"]["primary_D"]["radius"]-1.5*R)<1e-15
        assert r["verdict"]=="UNRESOLVED"
        # Missing shots / duplicate settings are hard failures, not silently dropped.
        data["blocks"][0]["settings"][0]["raw9"].pop()
        try: analyze(data,fixture=True)
        except AssertionError: pass
        else: raise AssertionError("Incomplete input not rejected")
        result=dict(status="PASS_NEW_READER_EXACT_ARITHMETIC_AND_INCOMPLETE_REJECTION",
                    origin="ARITHMETIC_FIXTURE_NOT_IBM",coordinate_radius=R,
                    primary_radius=1.5*R,real_data_read=0,QPU_jobs=0)
        with (HERE/"READER_FIXTURE_CHECK.json").open("x") as stream: json.dump(result,stream,indent=2);stream.write("\n")
        print(json.dumps(result,indent=2))

if __name__=="__main__": main()
