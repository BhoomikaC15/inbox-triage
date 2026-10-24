#!/usr/bin/env python3
"""Build one self-contained, Gmail-style HTML report from emails + predictions.

First screen: the count for each label. Click a label to open its list; click a
row to read the email, see the quote the label was based on, and any checks that
failed. Email text is inserted with textContent only (never as HTML), and the file
makes no network requests. Nothing is deleted, moved or sent.

Usage:
  python3 make_report.py emails.jsonl preds.jsonl --out report.html
  python3 make_report.py test.jsonl preds_tinker.jsonl --fallback --show-gold --model-name "Qwen3.5-4B (fine-tuned)"
"""
import argparse
import email.utils
import json
from datetime import date

from postprocess import apply as apply_fallback

LABELS = ["deadline", "needs_reply", "might_interest", "skippable"]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def nice_date(iso):
    d = date.fromisoformat(iso)
    return f"{d.strftime('%a')} {d.day} {d.strftime('%b')}"


def snippet(body):
    lines = [l.strip() for l in body.splitlines() if l.strip() and not l.lstrip().startswith(">")]
    text = " ".join(lines[1:] if lines and len(lines[0]) < 22 and len(lines) > 1 else lines)
    return text[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emails")
    ap.add_argument("preds")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--fallback", action="store_true", help="let code fill in dates the checker dropped")
    ap.add_argument("--show-gold", action="store_true", help="mark emails where the label differs from the gold label")
    ap.add_argument("--model-name", default="")
    ap.add_argument("--title", default="Inbox triage")
    args = ap.parse_args()

    emails = load(args.emails)
    preds = {}
    for p in load(args.preds):
        preds[p["id"]] = p  # last one wins
    items = []
    for e in emails:
        p = preds.get(e["id"])
        if p is None:
            continue
        if args.fallback:
            p = apply_fallback(e, p)
        name, addr = email.utils.parseaddr(e["from"])
        item = {
            "id": e["id"], "name": name or addr or e["from"], "addr": addr, "to": e.get("to", ""),
            "subject": e["subject"], "sent": e["sent"], "sentLabel": nice_date(e["sent"]),
            "body": e["body"], "snippet": snippet(e["body"]),
            "label": p["label"] if p["label"] in LABELS else "needs_reply",
            "deadline": p.get("deadline_date"),
            "deadlineLabel": nice_date(p["deadline_date"]) if p.get("deadline_date") else None,
            "evidence": p.get("evidence", ""), "flags": p.get("flags", []),
        }
        if args.show_gold and "label" in e:
            item["gold"] = e["label"]
        items.append(item)

    payload = {"items": items, "model": args.model_name, "generated": date.today().isoformat()}
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html_out = TEMPLATE.replace("__TITLE__", args.title).replace("__DATA__", data)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_out)
    counts = {k: sum(1 for i in items if i["label"] == k) for k in LABELS}
    print(f"Wrote {args.out} with {len(items)} emails: {counts}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#f6f8fc; --surface:#fff; --text:#1f1f1f; --muted:#5f6368; --line:#e3e6ea; --hover:#f2f6fc;
  --select:#d3e3fd; --focus:#0b57d0;
  --dl:#b3261e; --dl-bg:#fce8e6; --rp:#0b57d0; --rp-bg:#e8f0fe;
  --in:#146c2e; --in-bg:#e6f4ea; --sk:#5f6368; --sk-bg:#eef0f2; --mark:#fff3b0;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#1b1c1f; --surface:#25262a; --text:#e8eaed; --muted:#a6abb2; --line:#383a40; --hover:#2d2f35;
    --select:#33435e; --focus:#8ab4f8;
    --dl:#f2b8b5; --dl-bg:#4a2321; --rp:#aecbfa; --rp-bg:#26364f;
    --in:#a8dab5; --in-bg:#22392a; --sk:#b9bec5; --sk-bg:#34363b; --mark:#5c5320;
  }
}
:root[data-theme="dark"]{
  --bg:#1b1c1f; --surface:#25262a; --text:#e8eaed; --muted:#a6abb2; --line:#383a40; --hover:#2d2f35;
  --select:#33435e; --focus:#8ab4f8;
  --dl:#f2b8b5; --dl-bg:#4a2321; --rp:#aecbfa; --rp-bg:#26364f;
  --in:#a8dab5; --in-bg:#22392a; --sk:#b9bec5; --sk-bg:#34363b; --mark:#5c5320;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Roboto,"Segoe UI",system-ui,-apple-system,sans-serif}
button{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.top{display:flex;align-items:center;gap:16px;padding:10px 16px}
.brand{display:flex;align-items:center;gap:10px;font-size:20px;min-width:208px}
.brand svg{flex:none}
.search{flex:1;max-width:680px}
.search input{width:100%;height:46px;border:0;border-radius:24px;background:var(--hover);color:var(--text);padding:0 20px;font:inherit}
.search input::placeholder{color:var(--muted)}
.themebtn{margin-left:auto;border:1px solid var(--line);background:transparent;border-radius:18px;padding:6px 12px;cursor:pointer;color:var(--muted)}
.layout{display:flex;gap:8px;padding:0 16px 16px}
nav{width:240px;flex:none}
nav button{display:flex;align-items:center;gap:14px;width:100%;height:36px;border:0;border-radius:0 18px 18px 0;background:transparent;padding:0 16px;cursor:pointer;text-align:left}
nav button:hover{background:var(--hover)}
nav button[aria-current="true"]{background:var(--select);font-weight:700}
nav .n{margin-left:auto;color:var(--muted);font-size:12px}
nav button[aria-current="true"] .n{color:var(--text)}
.dot{width:12px;height:12px;border-radius:50%;flex:none}
.l-deadline{--c:var(--dl);--b:var(--dl-bg)} .l-needs_reply{--c:var(--rp);--b:var(--rp-bg)}
.l-might_interest{--c:var(--in);--b:var(--in-bg)} .l-skippable{--c:var(--sk);--b:var(--sk-bg)}
.dot{background:var(--c)}
main{flex:1;min-width:0;background:var(--surface);border-radius:16px;min-height:70vh;overflow:hidden}
.pad{padding:24px 28px}
h1{font-size:22px;font-weight:400;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 20px}
.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.card{border:1px solid var(--line);border-radius:16px;background:var(--surface);padding:18px 20px;text-align:left;cursor:pointer;border-top:5px solid var(--c)}
.card:hover{background:var(--hover)}
.card .num{font-size:44px;line-height:1.1;font-weight:400;color:var(--c)}
.card .nm{font-size:16px;font-weight:700;margin-top:2px}
.card .hint{color:var(--muted);margin:2px 0 10px}
.card .peek{font-size:13px;color:var(--muted);border-top:1px solid var(--line);padding-top:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .peek b{color:var(--text);font-weight:500}
.note{margin-top:18px;color:var(--muted);font-size:13px}
.head{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line)}
.back{border:0;background:transparent;border-radius:50%;width:36px;height:36px;cursor:pointer;font-size:18px}
.back:hover{background:var(--hover)}
.head .t{font-size:18px;font-weight:500}
.head .c{color:var(--muted)}
.row{display:flex;align-items:center;gap:12px;min-height:42px;padding:6px 16px;border-bottom:1px solid var(--line);cursor:pointer}
.row:hover{background:var(--hover);box-shadow:inset 3px 0 0 var(--c)}
.row .who{width:190px;flex:none;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row .msg{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row .msg span{color:var(--muted)}
.row .when{flex:none;font-size:12px;color:var(--muted);text-align:right}
.chip{display:inline-block;border-radius:8px;padding:1px 9px;font-size:12px;font-weight:700;background:var(--b);color:var(--c);white-space:nowrap}
.open{border-bottom:1px solid var(--line);padding:16px 20px 20px 40px;background:var(--surface)}
.open h2{font-size:20px;font-weight:400;margin:0 0 6px}
.meta{color:var(--muted);font-size:13px;margin-bottom:12px}
.why{background:var(--b);border-left:4px solid var(--c);border-radius:6px;padding:10px 14px;margin:0 0 14px}
.why .k{font-size:12px;font-weight:700;color:var(--c);text-transform:uppercase;letter-spacing:.04em}
.why q{display:block;margin-top:2px;font-style:italic}
.flag{display:inline-block;margin:6px 6px 0 0;padding:2px 8px;border:1px dashed var(--muted);border-radius:8px;font-size:12px;color:var(--muted)}
.body{white-space:pre-wrap;overflow-wrap:anywhere;max-width:72ch}
mark{background:var(--mark);color:inherit;border-radius:3px;padding:0 1px}
.gold{color:#b3261e;font-weight:700;font-size:12px;margin-left:8px}
.empty{padding:40px;text-align:center;color:var(--muted)}
@media (max-width:820px){
  .layout{flex-direction:column;padding:0 12px 12px}
  nav{width:auto;display:flex;gap:6px;overflow-x:auto;padding-bottom:4px}
  nav button{width:auto;border-radius:18px;white-space:nowrap;flex:none;border:1px solid var(--line)}
  .brand{min-width:0} .brand span{display:none}
  .cards{grid-template-columns:1fr}
  .row{flex-wrap:wrap} .row .who{width:100%} .pad{padding:18px 16px}
  .open{padding:14px 16px}
}
</style>
</head>
<body>
<div class="top">
  <div class="brand"><svg width="28" height="28" viewBox="0 0 24 24" aria-hidden="true"><rect x="2" y="5" width="20" height="14" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M3 7l9 6 9-6" fill="none" stroke="currentColor" stroke-width="2"/></svg><span>__TITLE__</span></div>
  <label class="search"><input id="q" type="search" placeholder="Search sender, subject or text" aria-label="Search emails"></label>
  <button class="themebtn" id="theme" type="button">Theme</button>
</div>
<div class="layout">
  <nav id="nav" aria-label="Labels"></nav>
  <main id="main"></main>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
var D = JSON.parse(document.getElementById('data').textContent);
var ITEMS = D.items;
var LABELS = [
 {k:'deadline', n:'Deadlines', h:'Things you must do by a date'},
 {k:'needs_reply', n:'Needs reply', h:'Someone is waiting for your answer'},
 {k:'might_interest', n:'Might interest you', h:'Optional, but possibly worth a look'},
 {k:'skippable', n:'Skippable', h:'Ads, receipts, alerts and FYIs'}
];
var FLAGS = {
 date_not_found_in_email:'The model’s date was not in the email, so it was dropped',
 date_resolved_by_code:'Date worked out by code from the email text',
 evidence_not_in_email:'The quoted line could not be matched to the email',
 deadline_without_valid_date:'Deadline label without a usable date',
 model_failed:'The model failed, so this email is shown to be safe',
 unparseable:'The model’s answer could not be read'
};
var state = {view:'overview', open:null, q:''};
var $ = function(id){return document.getElementById(id)};
function el(tag, cls, text){var e=document.createElement(tag); if(cls)e.className=cls; if(text!==undefined)e.textContent=text; return e;}
function byLabel(k){
  var r = ITEMS.filter(function(i){return i.label===k});
  if(k==='deadline') r.sort(function(a,b){return (a.deadline||'9999').localeCompare(b.deadline||'9999')});
  else r.sort(function(a,b){return b.sent.localeCompare(a.sent)});
  return r;
}
function meta(k){return LABELS.filter(function(l){return l.k===k})[0]}
function go(v){state.view=v; state.open=null; render(); $('main').scrollIntoView({block:'nearest'});}

function renderNav(){
  var nav = $('nav'); nav.textContent='';
  var all = el('button'); all.type='button';
  all.setAttribute('aria-current', String(state.view==='overview' && !state.q));
  all.appendChild(el('span','',  'Overview')); nav.appendChild(all);
  all.onclick=function(){$('q').value=''; state.q=''; go('overview')};
  LABELS.forEach(function(l){
    var b = el('button','l-'+l.k); b.type='button';
    b.setAttribute('aria-current', String(state.view===l.k && !state.q));
    b.appendChild(el('span','dot'));
    b.appendChild(el('span','',l.n));
    b.appendChild(el('span','n',String(byLabel(l.k).length)));
    b.onclick=function(){$('q').value=''; state.q=''; go(l.k)};
    nav.appendChild(b);
  });
}

function renderOverview(m){
  var pad = el('div','pad');
  pad.appendChild(el('h1','', ITEMS.length+' emails, sorted'));
  var sub = 'Pick a label to see what is inside.';
  pad.appendChild(el('p','sub',sub));
  var grid = el('div','cards');
  LABELS.forEach(function(l){
    var list = byLabel(l.k);
    var c = el('button','card l-'+l.k); c.type='button';
    c.appendChild(el('div','num',String(list.length)));
    c.appendChild(el('div','nm',l.n));
    c.appendChild(el('div','hint',l.h));
    var peek = el('div','peek');
    if(list.length){
      var first = list[0];
      if(l.k==='deadline' && first.deadlineLabel){ peek.appendChild(document.createTextNode('Next: ')); peek.appendChild(el('b','',first.deadlineLabel)); peek.appendChild(document.createTextNode(' – '+first.subject)); }
      else { peek.appendChild(document.createTextNode('Latest: ')); peek.appendChild(el('b','',first.subject)); }
    } else peek.textContent='Nothing here';
    c.appendChild(peek);
    c.onclick=function(){go(l.k)};
    grid.appendChild(c);
  });
  pad.appendChild(grid);
  var foot = 'Triage only. Nothing is deleted, moved or sent.' + (D.model? ' Labelled by '+D.model+'.':'');
  var flagged = ITEMS.filter(function(i){return i.flags && i.flags.length}).length;
  if(flagged) foot += ' '+flagged+' emails have a check note; open them to see why.';
  pad.appendChild(el('p','note',foot));
  m.appendChild(pad);
}

function highlight(container, text, quote){
  var idx = quote ? text.toLowerCase().indexOf(quote.toLowerCase()) : -1;
  if(idx<0){container.textContent=text; return;}
  container.appendChild(document.createTextNode(text.slice(0,idx)));
  container.appendChild(el('mark','',text.slice(idx, idx+quote.length)));
  container.appendChild(document.createTextNode(text.slice(idx+quote.length)));
}

function renderDetail(i){
  var box = el('div','open l-'+i.label);
  box.appendChild(el('h2','',i.subject));
  var mt = el('div','meta');
  mt.textContent = i.name + (i.addr? ' <'+i.addr+'>':'') + '  ·  ' + i.sentLabel;
  box.appendChild(mt);
  var why = el('div','why');
  why.appendChild(el('div','k','Why: '+meta(i.label).n+(i.deadlineLabel? ' · '+i.deadlineLabel:'')));
  why.appendChild(i.evidence ? el('q','',i.evidence) : el('span','', 'No supporting line was given.'));
  box.appendChild(why);
  var b = el('div','body'); highlight(b, i.body, i.evidence); box.appendChild(b);
  (i.flags||[]).forEach(function(f){ if(FLAGS[f]) box.appendChild(el('span','flag',FLAGS[f])); });
  if(i.gold && i.gold!==i.label) box.appendChild(el('div','gold','Gold label differs: '+meta(i.gold).n));
  return box;
}

function renderRow(i, m){
  var r = el('div','row l-'+i.label); r.tabIndex=0; r.setAttribute('role','button');
  r.setAttribute('aria-expanded', String(state.open===i.id));
  r.appendChild(el('div','who',i.name));
  var msg = el('div','msg'); msg.appendChild(document.createTextNode(i.subject)); msg.appendChild(el('span','',' – '+i.snippet)); r.appendChild(msg);
  var w = el('div','when');
  if(i.deadlineLabel) w.appendChild(el('span','chip',i.deadlineLabel)); else w.textContent=i.sentLabel;
  r.appendChild(w);
  function toggle(){state.open = state.open===i.id ? null : i.id; render();}
  r.onclick=toggle; r.onkeydown=function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle();}};
  m.appendChild(r);
  if(state.open===i.id) m.appendChild(renderDetail(i));
}

function renderList(m, list, title, k){
  var h = el('div','head l-'+(k||'skippable'));
  var back = el('button','back','←'); back.type='button'; back.setAttribute('aria-label','Back to overview');
  back.onclick=function(){$('q').value=''; state.q=''; go('overview')};
  h.appendChild(back);
  if(k) h.appendChild(el('span','dot'));
  h.appendChild(el('span','t',title)); h.appendChild(el('span','c',list.length+' emails'));
  m.appendChild(h);
  if(!list.length){m.appendChild(el('div','empty','Nothing here.')); return;}
  list.forEach(function(i){renderRow(i,m)});
}

function render(){
  renderNav();
  var m = $('main'); m.textContent='';
  if(state.q){
    var q = state.q.toLowerCase();
    var res = ITEMS.filter(function(i){return (i.name+' '+i.addr+' '+i.subject+' '+i.body).toLowerCase().indexOf(q)>=0});
    renderList(m, res, 'Search results', null); return;
  }
  if(state.view==='overview') renderOverview(m);
  else renderList(m, byLabel(state.view), meta(state.view).n, state.view);
}

$('q').addEventListener('input', function(e){state.q=e.target.value.trim(); state.open=null; render();});
var themes=['auto','light','dark'], ti=0;
$('theme').onclick=function(){ti=(ti+1)%3; var t=themes[ti]; if(t==='auto')document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme',t); this.textContent='Theme: '+t;};
render();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
