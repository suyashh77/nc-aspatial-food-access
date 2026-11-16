"""Builds Final_Presentation.html (same format as the UPL deck: 1920x1080 slides, arrow keys, N = speaker notes, F = full screen).

    python3 handoff/presentation/build_deck.py

Then the PDF:  google-chrome --headless --no-pdf-header-footer --print-to-pdf=Final_Presentation.pdf Final_Presentation.html
Numbers are typed in below from PIPELINE_GUIDE.md. After the human labels are in, update the slides marked
PROVISIONAL (search for "prov(") and rebuild.
"""
from pathlib import Path

from make_maps import diverge_legend, reach_map, score_map, stores_map

HERE = Path(__file__).resolve().parent
CSS = (HERE / "deck_style.css").read_text()
GREEN, GREY = "#008300", "#a9aeb5"
SLIDES: list[str] = []


def slide(sec, kick, title, body, tk="", notes="", tall=False, chip=""):
    chip_html = f'<span class="chip on" style="position:absolute;right:88px;top:74px;font-size:17px">{chip}</span>' if chip else ""
    tk_html = f'<div class="tk">{tk}</div>' if tk else ""
    notes = notes.replace('"', "&quot;")
    SLIDES.append(f'<section class="slide" data-sec="{sec}" data-notes="{notes}"><header class="hd"><p class="kick">{kick}</p>'
                  f'<h2>{title}</h2></header>{chip_html}<div class="bd{" tall" if tall else ""}">{body}</div>{tk_html}</section>')


def prov():
    return "Provisional · before hand labeling"


def bars(rows, vmax, label_w=260, val_w=150, h=34, color=GREEN):
    """rows: (label, sublabel, value, shown text)"""
    out = []
    for lab, sub, v, txt in rows:
        sub_html = f'<br><span style="font-size:17px;font-weight:400;color:#5b6067">{sub}</span>' if sub else ""
        out.append(f'<div style="display:grid;grid-template-columns:{label_w}px 1fr {val_w}px;gap:14px;align-items:center;margin:12px 0">'
                   f'<span style="font-size:23px;color:#111;font-weight:600;line-height:1.15">{lab}{sub_html}</span>'
                   f'<div style="height:{h}px;background:#eef0f2;border-radius:4px;overflow:hidden"><div style="height:{h}px;width:{100 * v / vmax:.1f}%;background:{color}"></div></div>'
                   f'<span style="font-size:26px;font-weight:700;color:#111;text-align:right">{txt}</span></div>')
    return "".join(out)


def pair(rows, vmax, la, lb, label_w=230, fmt="{:.2f}"):
    """rows: (label, a, b). Two bars per row: a grey, b green. Values are drawn by magnitude."""
    leg = (f'<div class="leg" style="margin-bottom:8px"><span><i style="background:{GREY}"></i>{la}</span>'
           f'<span><i style="background:{GREEN}"></i>{lb}</span></div>')
    out = [leg]
    for lab, a, b in rows:
        cells = []
        for v, c in ((a, GREY), (b, GREEN)):
            cells.append(f'<div style="display:grid;grid-template-columns:1fr 90px;gap:12px;align-items:center;margin:3px 0">'
                         f'<div style="height:24px;background:#eef0f2;border-radius:4px;overflow:hidden"><div style="height:24px;width:{100 * abs(v) / vmax:.1f}%;background:{c}"></div></div>'
                         f'<span style="font-size:22px;font-weight:700;color:#111;text-align:right">{fmt.format(v).replace("-", "−")}</span></div>')
        out.append(f'<div style="display:grid;grid-template-columns:{label_w}px 1fr;gap:16px;align-items:center;margin:10px 0">'
                   f'<span style="font-size:23px;color:#111;font-weight:600">{lab}</span><div>{"".join(cells)}</div></div>')
    return "".join(out)


def card(label, big, text="", inv=False, big_size=64):
    return (f'<div class="cd{" inv" if inv else ""}" style="padding:22px 26px"><p class="lb">{label}</p>'
            f'<p class="big" style="font-size:{big_size}px;line-height:1.05">{big}</p>'
            + (f'<p class="sm" style="margin-top:8px;color:#111">{text}</p>' if text else "") + "</div>")


def table(head, rows, cls="tb", font=24):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}" style="font-size:{font}px"><tr>{th}</tr>{tr}</table>'


def steps(items):
    return '<ul class="steps">' + "".join(f'<li><span class="k">{i + 1}</span><div><h3>{h}</h3><p>{p}</p></div></li>'
                                          for i, (h, p) in enumerate(items)) + "</ul>"


DIMS = ["Availability", "Accessibility", "Accommodation", "Affordability", "Acceptability"]

# ---------------------------------------------------------------- title + agenda
SLIDES.append('<section class="slide title-s" data-sec="0"><div class="glow"></div><div class="tl">'
              '<p class="kick">Undergraduate Research · NC State ISE</p>'
              '<h1>Food Access<br/>from Google Reviews</h1>'
              '<p class="date" style="font-size:34px;color:#2b2d31;margin-top:24px">Scaling Teresa Groton&#8217;s Raleigh method to all of North Carolina</p>'
              '<p class="by">Suyash Sapre  ·  Dr. Benjamin Rachunok&#8217;s group  ·  Jan–Nov 2026</p>'
              '<p class="date">Final presentation and handoff · November 2026</p></div></section>')

agenda = [("01", "Situation", "Introduction", ["The project at a glance", "Why distance isn't access", "Teresa's Raleigh pilot"], "3–5"),
          ("02", "Task", "Goal", ["Every NC grocery store, scored on five dimensions", "Reproduce Teresa first"], "6"),
          ("03", "Action", "What we did", ["Store list, scope, scraping", "Is it authoritative?", "Replicate, scale, improve", "Hand labeling"], "7–15"),
          ("04", "Result", "Results", ["What the reviews say", "Statewide and Raleigh gaps", "What we hand off"], "16–22"),
          ("05", "Reflection", "Limits and next steps", ["What's wrong with the data", "Lessons", "What's left"], "23–25")]
cells = []
for n, chip, h, items, sl in agenda:
    li = "".join(f'<li style="margin:0 0 10px;padding-left:16px;text-indent:-16px">· {x}</li>' for x in items)
    cells.append(f'<div class="cd ag" style="min-height:0;padding:24px 24px;display:flex;flex-direction:column">'
                 f'<div style="display:flex;justify-content:space-between;align-items:baseline"><span class="big" style="font-size:52px">{n}</span><span class="chip">{chip}</span></div>'
                 f'<h3 style="font-size:30px;margin-top:8px;line-height:1.15">{h}</h3><ul style="list-style:none;padding:0;margin:14px 0 0;font-size:21px;line-height:1.3;color:#2b2d31">{li}</ul>'
                 f'<span class="rg" style="margin-top:auto">Slides {sl}</span></div>')
SLIDES.append('<section class="slide" data-sec="0"><header class="hd"><p class="kick">Overview · STAR-R</p><h2>Agenda</h2></header>'
              '<div class="bd tall" style="top:210px"><div class="agrid" style="grid-template-columns:repeat(5,minmax(0,1fr));gap:22px;align-items:stretch">'
              + "".join(cells) + "</div></div></section>")

# ---------------------------------------------------------------- 1 situation
# ---------------------------------------------------------------- the project at a glance
flow = [("Store list", "4 sources → 36,820 candidates"), ("Scope", "~300 rulings → 4,442 stores"),
        ("Reviews", "1.94M scraped, frozen Oct 24"), ("Authoritative?", "99.6% vs capped counts"),
        ("Replicate", "Teresa&#8217;s Raleigh ranks: ρ 0.90–0.95"), ("Pipeline A", "her method on all of NC"),
        ("Pipeline B", "five tested fixes"), ("Hand labels", "Wednesday · then one script")]
chev = "".join(f'<div class="{"on" if i == 7 else ""}" style="font-size:21px;padding-left:{30 if i == 0 else 44}px;{"margin-right:0" if i == 7 else ""}">{h}</div>' for i, (h, _) in enumerate(flow))
desc = "".join(f'<p class="sm" style="font-size:20px;color:#2b2d31;padding:0 10px 0 {14 if i == 0 else 28}px">{d}</p>' for i, (_, d) in enumerate(flow))
body = (f'<div class="chev">{chev}</div><div style="display:grid;grid-template-columns:repeat(8,1fr);margin-top:14px">{desc}</div>'
        '<div class="g" style="grid-template-columns:1fr 1fr;gap:24px;margin-top:44px">'
        '<div class="cd" style="padding:20px 24px"><p class="lb">Steps 1–4 · build the data</p><p class="sm" style="color:#111">Find every grocery store in NC, collect every Google review we can reach, and write down every gap.</p></div>'
        '<div class="cd inv" style="padding:20px 24px"><p class="lb">Steps 5–8 · measure access</p><p class="sm" style="color:#111">Reproduce Teresa first, then run her method statewide, then fix what testing found. People check the labels last.</p></div></div>')
slide(1, "Introduction", "The project at a glance", body,
      "Data first, then replication, then improvement. Only the hand labels are left.",
      "Use this as the map for the talk. Every later slide is one of these boxes.")

five = [("Availability", "Is the food there?", "Stock, selection, empty shelves"),
        ("Accessibility", "Can I get there?", "Distance, parking, the bus"),
        ("Accommodation", "Does it fit my life?", "Hours, SNAP/WIC, pickup, checkout"),
        ("Affordability", "Can I pay for it?", "Prices, deals, value"),
        ("Acceptability", "Does it meet my standards?", "Freshness, cleanliness, staff")]
body = ('<div class="g" style="grid-template-columns:560px 1fr;gap:48px;align-items:center">'
        '<div class="col" style="gap:20px"><p class="stmt" style="font-size:40px">A food desert is usually drawn with <span class="b">distance</span>: more than a mile from a supermarket.</p>'
        '<p class="sm" style="font-size:27px">But a store next door doesn&#8217;t help if it&#8217;s too expensive, dirty, empty, or closed when you get off work.</p>'
        '<p class="xs mu">Penchansky &amp; Thomas (1981), adapted to food by Caspi et al. (2012)</p></div>'
        '<div class="g" style="grid-template-columns:1fr;gap:12px">'
        + "".join(f'<div class="cd" style="padding:16px 24px;display:grid;grid-template-columns:270px 1fr;gap:16px;align-items:baseline"><b style="font-size:28px;color:#111">{a}</b>'
                  f'<span style="font-size:24px;color:#111">{q} <span class="mu" style="font-size:21px">· {e}</span></span></div>' for a, q, e in five)
        + "</div></div>")
slide(1, "Introduction", "Why distance isn&#8217;t access", body,
      "Access has five dimensions. Distance measures one of them.",
      "Start here: the usual food desert map is distance only. The five A's are the framework Teresa used. Each one is something people actually write about in reviews.")

body = ('<div class="g" style="grid-template-columns:1fr 600px;gap:44px;align-items:center"><div>'
        '<p class="lb">How one review becomes part of a score</p>'
        + steps([("Read the text", "Each review becomes 384 numbers (a sentence model)"),
                 ("Group similar reviews", "Reviews cluster into topics: &#8220;parking&#8221;, &#8220;prices&#8221;, &#8220;rude cashiers&#8221;"),
                 ("Name the dimension", "Each topic matches the dimension whose definition it sounds most like"),
                 ("Good or bad from the stars", "1★ = −1, 2–4★ = 0, 5★ = +1"),
                 ("Store, then neighborhood", "Average per store, pooled; nearby stores weighted into each census tract")])
        + '</div><div class="col" style="gap:18px">' + card("Teresa&#8217;s pilot", "49 stores", "about 25,000 Google reviews, Raleigh") +
        card("Her finding", "Reviews work", "They sort into the five dimensions, and tract scores differ by race and income", inv=True) + "</div></div>")
slide(1, "Introduction", "Teresa&#8217;s idea: reviews are a survey nobody had to run", body,
      "The text decides which dimension a review is about. The stars decide good or bad.",
      "Teresa Groton's master's thesis. Our job was to take this from 49 stores to the whole state.")

# ---------------------------------------------------------------- 2 task
goals = [("01", "Every store", "A complete list of NC grocery stores, with a written rule for what counts"),
         ("02", "Every review", "Every Google review we can reach for each store, checked against Google&#8217;s counts"),
         ("03", "Reproduce first", "Show our data gives Teresa&#8217;s Raleigh results before changing anything"),
         ("04", "Then improve", "Fix what testing finds, one change at a time, each measured against her method")]
body = ('<div class="cd inv" style="padding:16px 24px;margin-bottom:22px"><p style="font-size:30px;color:#111;line-height:1.3"><b>Goal:</b> score every grocery store and census tract in North Carolina on the five dimensions of access, using Teresa&#8217;s method.</p></div>'
        '<div class="g" style="grid-template-columns:repeat(4,1fr);gap:20px">'
        + "".join(f'<div class="cd" style="padding:30px 30px;min-height:330px"><p class="n">{n}</p><h3 style="font-size:36px;margin-top:6px">{h}</h3><p class="sm" style="margin-top:12px;font-size:26px">{p}</p></div>' for n, h, p in goals)
        + "</div>")
slide(2, "Goal", "What were we trying to do?", body,
      "A method that works for 49 stores has to be rebuilt to work for 4,442. Reproducing her results is what makes the rest credible.",
      "Four parts. The first two are data. The third is the trust check. The fourth is the new research.")

# ---------------------------------------------------------------- 3 action
body = ('<div class="g" style="grid-template-columns:1fr 560px;gap:44px;align-items:center"><div class="cd" style="padding:24px 28px">'
        '<p class="lb">From raw records to the final store list</p>'
        + bars([("Raw records", "SNAP · Overture · OSM · project data", 91461, "91,461"),
                ("Food retail frame", "matched across sources (geomatch)", 36820, "36,820"),
                ("Probed on Google", "established, bricks and mortar", 16751, "16,751"),
                ("Grocery stores", "after scope rules and manual checks", 4442, "4,442")], 91461, label_w=330)
        + '</div><div class="col" style="gap:18px">'
        + card("Why four sources", "No list is complete", "SNAP has good addresses but only SNAP stores. Overture has clean names but stale records. OpenStreetMap finds small ethnic and rural stores.")
        + card("The first list kept only", "14–38%", "of candidates, depending on area: worst in rural eastern NC. It would have drawn food deserts that were holes in the list.", inv=True)
        + "</div></div>")
slide(3, "Step 1 · Store list", "How did we find every grocery store?", body,
      "We rebuilt the list from four sources because the first one invented food deserts.",
      "The spring list came from Overture alone. In September we rebuilt it. geomatch is our own matching tool, on GitHub.")

body = ('<div class="g" style="grid-template-columns:1fr 640px;gap:44px;align-items:center"><div class="cd" style="padding:24px 28px">'
        '<p class="lb">The 4,442 stores by Google category</p>'
        + bars([("Grocery store", "", 1897, "1,897"), ("Dollar store", "Dollar General, Family Dollar", 1001, "1,001"),
                ("Supermarket", "", 212, "212"), ("Department store", "Walmart, Target", 174, "174"),
                ("Everything else", "butchers, seafood, ethnic grocers, health food…", 1158, "1,158")], 1897, label_w=300)
        + '</div><div class="col" style="gap:14px">'
        '<div class="cd" style="padding:20px 24px"><p class="lb">In</p><p class="sm" style="color:#111">Supermarkets, grocery chains, dollar stores, independent and ethnic grocers, butchers, co-ops. Any chain with more than five NC stores.</p></div>'
        '<div class="cd" style="padding:20px 24px"><p class="lb">Out</p><p class="sm" style="color:#111">Gas stations, convenience stores, farmers&#8217; markets, produce stands, restaurants, closed stores.</p></div>'
        '<div class="cd inv" style="padding:20px 24px"><p class="lb">Nothing deleted</p><p class="sm" style="color:#111">4,736 out-of-scope, 67 closed and 221 duplicate stores each sit in their own database, with a reason.</p></div>'
        "</div></div>")
slide(3, "Step 2 · Scope", "What counts as a grocery store?", body,
      "About 300 category rulings by hand. Dollar stores are a quarter of the list, on purpose.",
      "Dollar stores are in because in much of rural NC they are where people buy food. A study can split them out by category.")

svg, cnt = stores_map()
leg = "".join(f'<div style="display:flex;align-items:center;gap:12px;margin:10px 0;font-size:24px;color:#111"><i style="width:18px;height:18px;border-radius:50%;background:{c};display:inline-block"></i>{lab}<b style="margin-left:auto">{cnt[k]:,}</b></div>'
              for k, c, lab in (("grocery", "#2a78d6", "Grocery stores and supermarkets"), ("dollar", "#eb6834", "Dollar stores"), ("other", "#1baf7a", "Everything else")))
body = (f'<div class="g" style="grid-template-columns:1180px 1fr;gap:44px;align-items:center"><div>{svg}</div>'
        f'<div class="col" style="gap:18px"><div class="cd" style="padding:20px 24px"><p class="lb">Each dot is a store</p>{leg}</div>'
        + card("Coverage", "All 100 counties", "590 cities and towns. Dollar stores (orange) are spread through the rural counties.")
        + f'<p class="xs mu">{cnt["left_off"]} stores are left off: one has no location and the rest have addresses outside NC (a data error; see the limits slide).</p>'
        + "</div></div>")
slide(3, "Step 2 · Scope", "Where are the 4,442 stores?", body,
      "Supermarkets cluster in the cities. Across rural NC, many of the stores are dollar stores.",
      "County lines in light gray. Locations from Google; one farm with a pin outside NC is left off.")

body = ('<div class="g" style="grid-template-columns:600px 1fr;gap:44px;align-items:center"><div class="col" style="gap:18px">'
        + card("Reviews collected", "1,941,538", "977,398 with text · 590 cities · all 100 counties", inv=True, big_size=76)
        + card("Collection closed", "Oct 24, 2026", "Frozen database with a checksum (freeze v2)")
        + '</div><div><p class="lb">What made the scraper trustworthy</p>'
        + steps([("Open the store by its exact Google ID", "The first version searched by name. A Sam&#8217;s Club in Hendersonville came back as one in Raleigh."),
                 ("Check every scroll actually moved", "A scroll that silently does nothing looks exactly like the end of the list"),
                 ("One scraper per machine", "Google limits by IP address. NC State Linux machines, plus ISE 408 students early on"),
                 ("Never delete", "Suspect reviews are set aside with a reason, so any decision can be undone")])
        + "</div></div>")
slide(3, "Step 3 · Reviews", "How did we collect the reviews?", body,
      "A September audit found wrong-store scrapes. We fixed the code and the data before collecting the rest.",
      "Python and Playwright driving a headless browser, no paid API. The engineering lessons are in backup.")

body = ('<div class="g" style="grid-template-columns:repeat(3,1fr);gap:24px;margin-bottom:26px">'
        + card("Against Google&#8217;s counts, capped at 1,500", "99.6%", "4,400 stores complete under a written rule", inv=True)
        + card("Against Google&#8217;s true counts", "87.9%", "The gap is big stores: we hold their newest reviews")
        + card("Set aside, not deleted", "29,321", "reviews that belonged to another store or a collision")
        + "</div>"
        + table(["Store size (Google count)", "Complete if missing at most", "Stores short"],
                [["100 or under", "10", ""], ["101–500", "20", "42 (21–25 each)"], ["501–1,500", "50", ""], ["over 1,500", "100, against a target of 1,500", ""]], font=24))
slide(3, "Step 4 · Authoritative?", "Is this every review of every store?", body,
      "Close. Where it isn&#8217;t, the gap is written down store by store.",
      "Google lists some reviews it won't show. The 42 short stores were rescraped repeatedly. Not done: comparing the store list with Census county counts.")

body = ('<div class="g" style="grid-template-columns:1fr 560px;gap:44px;align-items:center"><div class="cd" style="padding:24px 28px">'
        '<p class="lb">Rank agreement with Teresa&#8217;s store scores · her 42 matched stores</p>'
        + pair([(d, a, b) for d, a, b in zip(DIMS, [0.17, 0.40, 0.43, 0.59, 0.93], [0.90, 0.95, 0.93, 0.91, 0.95])], 1.0,
               "Model refit on all of NC", "Her Raleigh model, our newer reviews")
        + '</div><div class="col" style="gap:18px">'
        + card("Her model on our data", "ρ 0.90–0.95", "Average scores match too: Affordability 0.52 vs her 0.53", inv=True)
        + card("Her Table 5.3", "+1.65 vs +1.64", "Affordability coefficient rebuilt from her own store scores. Her tract list is unknown, so this part is approximate.")
        + "</div></div>")
slide(3, "Step 5 · Replicate", "Does our data reproduce Teresa&#8217;s Raleigh results?", body,
      "Yes, with her model. A model refit on NC agrees with her only on Acceptability: it measures the other four differently.",
      "This is the trust check. The refit result is a finding about the method, not a failure: a new topic model picks different reviews for the small dimensions.")

body = ('<div class="g" style="grid-template-columns:1fr 1fr;gap:40px">'
        '<div><p class="lb">Pipeline A: Teresa&#8217;s method at NC scale</p>'
        + table(["Stage", "What changed, and why"],
                [["Clean, embed", "14 NC chain names added to her stopword list"],
                 ["Reduce (UMAP)", "Runs on the GPU: the CPU version needs more memory than the VM has"],
                 ["Topics", "Min cluster 120 → 2,000: 83 topics, coherence 0.529 vs her 0.528"],
                 ["Match to dimensions", "None. Her first-person definitions reproduce her numbers to 0.002"],
                 ["Store and tract scores", "None. Same model, same settings"]], font=22)
        + '</div><div class="col" style="gap:18px">'
        + card("Two versions", "Refit · Frozen", "Refit: topics learned from NC reviews. Frozen: her Raleigh model applied to NC reviews.")
        + card("Reviews scored", "975,388", "every written review the cleaning step doesn&#8217;t empty", inv=True)
        + card("Ten decision tests", "D1–D10", "Each open choice tested: which definitions, how many topics, reassign outliers, translate or not…")
        + "</div></div>")
slide(3, "Step 6 · Pipeline A", "Teresa&#8217;s method, changed only where scale forces it", body,
      "Every change from her method is forced by size, and each one is written down.",
      "D1: only her first-person definition text reproduces her saved numbers. D3: 44, 83 or 168 topics barely change results; 83 is half the hand-check work of 168.")

body = table(["Problem found in A", "Fix in Pipeline B", "Result"],
             [["One topic per review, but 79% of longer reviews cover 2+ subjects", "Read reviews sentence by sentence", "Finds 70% of labels, not 45%"],
              ["Her definitions put selection under Acceptability", "Definitions written from the raters&#8217; rules", "Availability recall 0.50 → 0.81"],
              ["A 3★ &#8220;great selection, high prices&#8221; scores 0 on both", "Good or bad read from each sentence&#8217;s text", "84% right (stars: 64%)"],
              ["One 5★ review can top a ranking", "Better pooling: count model, county and store-type levels", "Half the error for stores with ≤5 reviews"],
              ["exp(−miles) is an urban rule", "Real drive times on OpenStreetMap roads, 16-minute markets", "Shows who has no store in reach"]], font=23)
slide(3, "Step 7 · Pipeline B", "What did testing find, and how did we fix it?", body,
      "Each fix is one switch, scored against Teresa&#8217;s method on the same yardsticks.",
      "Caveat to say out loud: the accuracy numbers use a stand-in answer key labeled by Claude with the same rules as B's definitions, so they flatter B until people relabel.",
      chip=prov())

body = ('<div class="g" style="grid-template-columns:1fr 600px;gap:44px;align-items:center"><div>'
        + steps([("Label reviews", "260 reviews, 5 people each. For each dimension: not discussed, positive, negative or mixed. This is the answer key."),
                 ("Check topics", "83 topics in about 25 groups. Mark vague ones, score each group 0–3 per dimension. This decides which reviews count."),
                 ("Import and rerun", "One script imports the labels, takes the majority vote and reruns everything: about 1.5 hours.")])
        + '</div><div class="col" style="gap:18px">'
        + card("The biggest single call", "Topic 0", "58,000 reviews about &#8220;the grocery store&#8221;. Generic label, but much of it is about selection: Availability&#8217;s best evidence.", inv=True)
        + card("Tested", "Nov 15", "The whole after-labeling path ran end to end on fake labels from five raters")
        + "</div></div>")
slide(3, "Hand labeling", "What only people can do", body,
      "Until people label, every score is provisional. After, one command produces the final numbers.",
      "Scheduled for Wednesday. Suyash runs the session on the labeling page and pulls the results. Caroline's guide has the raters' rules.")

# ---------------------------------------------------------------- 4 results
body = ('<div class="g" style="grid-template-columns:1fr 560px;gap:44px;align-items:center"><div class="cd" style="padding:24px 28px">'
        '<p class="lb">Share of written reviews that count toward each dimension</p>'
        + pair([(d, a, b) for d, a, b in zip(DIMS, [11, 5, 6, 18, 58], [48, 15, 36, 27, 66])], 70, "Pipeline A", "Pipeline B", fmt="{:.0f}%")
        + '</div><div class="col" style="gap:18px">'
        + card("Reviews counting toward any dimension", "69% → 90%", "Sentences pull parking, hours and selection out of reviews that are mostly about something else", inv=True)
        + card("Median reviews per store, Pipeline A", "60 vs 5", "Acceptability vs Accommodation. Most reviews are about staff, cleanliness and freshness.")
        + "</div></div>")
slide(4, "Results · Coverage", "What do people write about?", body,
      "Acceptability dominates. Reading sentences triples the evidence for the smaller dimensions.",
      "Watch Availability in B: the new definitions send product topics like 'great produce section' there. The human answer key decides if that's right.", chip=prov())

body = ('<div class="g" style="grid-template-columns:1fr 520px;gap:44px;align-items:center"><div class="cd" style="padding:24px 28px">'
        '<p class="lb">Coefficient on % Black (share), statewide, controlling for income · below zero = worse</p>'
        + pair([(d, a, b) for d, a, b in zip(DIMS, [-0.36, -0.61, -0.37, -0.24, -0.71], [-0.68, -1.69, -1.02, -1.31, -1.13])], 1.8,
               "A: stores within ~1 mile", "B: typical store within 16 minutes")
        + '</div><div class="col" style="gap:18px">'
        + card("Distance to the nearest store", "Closer", "Tracts with a higher Black share are nearer a store (−1.03), the reverse of Raleigh")
        + card("Quality of reachable stores", "Lower", "on every dimension, in both pipelines. B&#8217;s gaps are larger, Affordability most of all.", inv=True)
        + '<p class="xs mu">2,624 tracts, population-weighted. R² 0.01–0.08. A and B tract scores are standardized differently: compare direction, not exact size. The best reachable store is higher in these tracts.</p>'
        + "</div></div>")
slide(4, "Results · Statewide", "Closer to a store, but to worse stores", body,
      "Distance says these neighborhoods are well served. The reviews say the stores they reach are worse.",
      "Access to a good store isn't the gap. The typical store people use is. R-squared is small: race and income explain little of what's reachable.", chip=prov())

svg, ntr = score_map("Acceptability")
body = (f'<div class="g" style="grid-template-columns:1180px 1fr;gap:44px;align-items:center"><div>{svg}</div>'
        f'<div class="col" style="gap:18px"><div class="cd" style="padding:20px 24px"><p class="lb">Each dot is a census tract</p>'
        f'<p class="sm" style="color:#111;margin-bottom:12px">Score of the typical store within a 16-minute drive, Acceptability (staff, cleanliness, freshness)</p>{diverge_legend()}</div>'
        + card("Why Acceptability", "Most evidence", "The median store has 60 reviews on it. The other four dimensions look patchier.")
        + f'<p class="xs mu">Pipeline B, stores weighted by foot traffic. {ntr:,} tracts with a store in reach, drawn at their centers. Rural tracts reach few stores, so one store can set their color.</p>'
        + "</div></div>")
slide(4, "Results · Map", "What the reviews say about nearby stores", body,
      "Neighboring tracts in the same metro can reach very different stores. Distance alone can&#8217;t show that.",
      "Point to Charlotte, the Triangle and the Triad: red and blue tracts sit next to each other. That is the gap the regressions measure.", chip=prov())

svg, rr = reach_map()
leg = ('<div style="display:flex;align-items:center;gap:12px;margin:8px 0;font-size:22px;color:#111"><i style="width:12px;height:12px;border-radius:50%;background:#c9cdd2;display:inline-block"></i>A study store within 16 minutes</div>'
       '<div style="display:flex;align-items:center;gap:12px;margin:8px 0;font-size:22px;color:#111"><i style="width:18px;height:18px;border-radius:50%;background:#eb6834;display:inline-block"></i>None within 16 minutes</div>'
       '<div style="display:flex;align-items:center;gap:12px;margin:8px 0;font-size:22px;color:#111"><i style="width:18px;height:18px;border-radius:50%;background:#eb6834;border:2px solid #111;display:inline-block;box-sizing:border-box"></i>None within 32 minutes</div>')
body = (f'<div class="g" style="grid-template-columns:1180px 1fr;gap:44px;align-items:center"><div>{svg}</div>'
        f'<div class="col" style="gap:18px"><div class="cd" style="padding:20px 24px"><p class="lb">Each dot is a census tract</p>{leg}</div>'
        + card("No store within 16 minutes", f'{rr["people16"]:,}', f'people in {rr["tracts_none16"]} tracts', inv=True, big_size=56)
        + card("None within 32 minutes", f'{rr["people32"]:,}', f'people in {rr["tracts_none32"]} tracts', big_size=56)
        + '<p class="xs mu">Real drive times on OpenStreetMap roads, free-flow. Only NC stores count, so tracts on the state line look worse than they are.</p>'
        + "</div></div>")
slide(4, "Results · Reach", "Who has no grocery store within reach?", body,
      "Half a million North Carolinians have no study store within a 16-minute drive.",
      "Distance still matters in rural NC. The review scores add a second layer on top of it, they don't replace it.")

body = ('<div class="g" style="grid-template-columns:1fr 560px;gap:44px;align-items:center"><div>'
        + table(["Wake County · 8-minute markets", "Teresa&#8217;s method (A)", "Improved (B)"],
                [["Availability", "−2.00*", "−2.03*"], ["Accessibility", "−0.68", "−0.94"], ["Accommodation", "−1.36*", "−1.63*"],
                 ["Affordability", "−1.79*", "−0.59"], ["Acceptability", "−1.44*", "−1.30*"]], cls="tb num", font=26)
        + '<p class="xs mu" style="margin-top:10px">* p &lt; 0.05. Coefficient on % Black (share), 211 tracts, controlling for income.</p></div>'
        '<div class="col" style="gap:18px">'
        + card("Why 8 minutes", "105 → 21 stores", "Median stores in reach: 16 minutes covers most of the metro", big_size=44)
        + card("Both methods agree", "≈ 0.2 SD lower", "Availability, tract 10 points more Black (0.13–0.16 SD on Accommodation, Acceptability)", inv=True, big_size=44)
        + card("Teresa&#8217;s Affordability +1.64", "−0.41 in A, −0.05 in B", "with every Wake store in. It depends on her 49 stores.", big_size=44)
        + "</div></div>")
slide(4, "Results · Raleigh", "Raleigh: the gaps show within a short drive", body,
      "In Wake County, both methods find lower Availability, Accommodation and Acceptability where the Black share is higher.",
      "Affordability depends on the method, so don't lead with it. Without foot-traffic weights, the gaps are larger and Accessibility becomes significant too.", chip=prov())

body = ('<div class="g" style="grid-template-columns:1fr 1fr;gap:40px;align-items:start">'
        '<div class="cd" style="padding:26px 30px"><p class="lb">How stores perform</p><p class="big" style="font-size:72px">2 dimensions</p>'
        '<p class="sm" style="color:#111;margin-top:10px">Overall quality (clean, staffed, stocked) and price. A store that does well on one access topic does well on nearly all of them, except price.</p></div>'
        '<div class="cd inv" style="padding:26px 30px"><p class="lb">What people talk about</p><p class="big" style="font-size:72px">14–15 themes</p>'
        '<p class="sm" style="color:#111;margin-top:10px">Mostly product mix. The strongest theme the five A&#8217;s don&#8217;t name: <b>checkout</b> (self-checkout, registers, lines). Then refunds and service, and the trip itself (parking, pickup, crowds).</p></div>'
        '</div><p class="sm" style="margin-top:28px;font-size:26px;color:#111">Method: group topics by how they rise and fall together across 1,803 stores, the way the Big Five personality traits were found. Keep only factors that return in 50 bootstrap samples.</p>')
slide(4, "Results · Dimensions", "Are there really five dimensions?", body,
      "For scores, the five carry about two independent pieces of information. Checkout is the leading candidate for a sixth.",
      "The raters' 'other subject' write-ins can settle the sixth-dimension question.", chip=prov())

delivered = [("Store list", "4,442 stores, addresses, Google IDs, categories"),
             ("Scores", "Every store and tract, both pipelines, with intervals"),
             ("Every review", "1.94M rows with store, text and labels"),
             ("Database", "The frozen SQLite database and side databases"),
             ("Labeling page", "Shareable, 5 raters per review, owner exports"),
             ("One script", "Imports labels and reruns everything"),
             ("Three guides", "For Dr. Rachunok, for Caroline, for Claude"),
             ("Code", "Both pipelines, 14 tests, pinned versions")]
body = ('<div class="g" style="grid-template-columns:repeat(4,1fr);gap:18px">'
        + "".join(f'<div class="cd" style="padding:22px 24px;min-height:190px"><h3 style="font-size:28px;color:#111">{h}</h3><p class="sm" style="margin-top:10px;font-size:22px">{p}</p></div>' for h, p in delivered)
        + "</div>")
slide(4, "Results · Handoff", "What&#8217;s in the handoff?", body,
      "The documents, slides and data are in one Google Drive folder. The code is on GitHub.",
      "Licensed foot-traffic data stays on the lab server. Large regenerable files (embeddings) stay there too.")

# ---------------------------------------------------------------- 5 reflection
issues = [("Answer key is a stand-in", "Accuracy numbers measure agreement with Claude until people label"),
          ("Some stores don&#8217;t belong", "3 have addresses outside NC; 37 are listed by Google as restaurants"),
          ("Small samples", "Median store: 5 reviews on Accommodation, 7 on Accessibility"),
          ("Big stores hold newest reviews only", "257 stores, a quarter of all reviews"),
          ("Dates are vague", "&#8220;3 years ago&#8221; · good to about a year"),
          ("State lines", "Only NC stores: border tracts look worse than they are")]
body = ('<div class="g" style="grid-template-columns:repeat(3,1fr);gap:20px">'
        + "".join(f'<div class="cd" style="padding:24px 26px;min-height:200px"><p class="n">{i + 1:02d}</p><h3 style="font-size:28px;color:#111;margin-top:6px">{h}</h3><p class="sm" style="margin-top:10px;font-size:23px">{p}</p></div>' for i, (h, p) in enumerate(issues))
        + "</div>")
slide(5, "Limits", "What&#8217;s wrong with the data?", body,
      "Every limit is written down, and most are flagged in the files, so a study can filter or adjust for it.",
      "The full list is in the report for Dr. Rachunok. Others worth mentioning if asked: reviewers aren't a random sample of shoppers, reviewer names aren't IDs, and short non-English reviews aren't translated.")

body = table(["Challenge", "What it meant", "Lesson"],
             [["The first store list came from one source", "It would have drawn food deserts that were holes in the list", "The frame decides everything downstream. Check coverage before collecting"],
              ["The scraper trusted names", "Wrong-store reviews got into the data", "Verify by identity, not by name"],
              ["Scrapes that &#8220;finished&#8221; early", "Silent failures look exactly like success", "Check that every step actually did something"],
              ["Teresa&#8217;s method is a few notebooks", "Unstated choices (units, tract list) had to be reverse-engineered", "Replicate first, then improve, and write down every change"],
              ["Everyone wants new results", "Improvements made before the baseline can&#8217;t be judged", "Keep the faithful version next to the improved one"]], font=22)
slide(5, "Lessons", "What got in the way, and what did I learn?", body,
      "Most of the hard work was making the data trustworthy, not the modeling.",
      "If asked what I'd do differently: build the four-source store list first, and test the scraper on known stores before scaling up.")

body = ('<div class="g" style="grid-template-columns:1fr 1fr;gap:40px"><div><p class="lb">This week</p>'
        + steps([("Hand labeling (Wednesday)", "Suyash runs the session and pulls the results"),
                 ("Run one script", "About 1.5 hours. Scores lose the &#8220;provisional&#8221; label"),
                 ("Update the numbers", "Pipeline guide and slides marked provisional")])
        + '</div><div><p class="lb">After</p>'
        + steps([("Ask Teresa", "Her tract list, her store set for distance, raw vs pooled z"),
                 ("Check the store list against Census counts", "The best remaining test of completeness"),
                 ("Add stores across the state line", "So border tracts are scored fairly"),
                 ("Test checkout as a sixth dimension", "From the raters&#8217; write-ins")])
        + "</div></div>")
slide(5, "Next steps", "What&#8217;s left?", body,
      "One labeling session and one script stand between today and final numbers.",
      "Recommendation for the headline: Pipeline B as the main result, Pipeline A beside it as the replication.")

SLIDES.append('<section class="slide q-s" data-sec="0"><div class="tl"><h1>Questions?</h1></div></section>')

# ---------------------------------------------------------------- backup
body = table(["Test", "Question", "Answer"],
             [["D1", "Which definition text did Teresa use?", "First-person: reproduces her numbers to 0.002, 9/9 of her hand-scored topics"],
              ["D2", "Is one topic per review enough?", "Rankings mostly survive (ρ ≥ 0.85); sentences find a third more labels → B"],
              ["D3", "44, 83 or 168 topics?", "Results barely move (F1 0.59–0.60). Keep 83"],
              ["D4", "Reassign outlier reviews to a topic?", "Yes: triples coverage, ranks barely change"],
              ["D5", "Merge near-duplicate topics for the hand check?", "Yes: halves the work, 20 of 23 groups agree"],
              ["D6", "Frozen Raleigh model or NC refit?", "Report both: frozen for Raleigh, refit for NC stores"],
              ["D7", "Translate non-English reviews?", "No: 99.5% of 8+ word reviews are English"],
              ["D8", "Does the replication hold after pooling?", "Yes: ρ 0.90–0.95"],
              ["D9", "Her Table 5.3 units and tracts?", "Share and $10k; tract list unknown"],
              ["D10", "Where do definitions sit among the topics?", "Availability&#8217;s nearest reviews are in Topic 0"]], font=21)
slide(9, "Backup · Method", "The ten decision tests", body, notes="Results in NC Pipeline/decisions/out/. Each is written up in PIPELINE_GUIDE.md section 4.")

body = ('<div class="g" style="grid-template-columns:1fr 1fr;gap:40px;align-items:center"><div class="cd" style="padding:24px 28px">'
        '<p class="lb">Median reviews behind each method&#8217;s top-10 stores</p>'
        + pair([(d, a, b) for d, a, b in zip(DIMS, [2, 1, 1, 2.5, 2.5], [29.5, 33, 13.5, 39.5, 123])], 123, "Teresa&#8217;s pooling (A)", "Better pooling (B5)", fmt="{:g}")
        + '</div><div><p class="sm" style="font-size:26px;color:#111">A store whose reviews on a dimension are all 5★ has a standard error of zero. Teresa&#8217;s code floors it at one in a million, so the model treats the store as perfectly measured and doesn&#8217;t shrink it.</p>'
        '<p class="sm" style="font-size:26px;color:#111;margin-top:18px">B5 models the counts of low, neutral and high reviews and adds county and store-type levels. On held-out reviews it halves A&#8217;s error for stores with five or fewer reviews.</p></div></div>')
slide(9, "Backup · Pooling", "Why A&#8217;s top stores have one or two reviews", body,
      "A keeps the floor because it&#8217;s Teresa&#8217;s method. B fixes it.", chip=prov())

body = ('<div class="g" style="grid-template-columns:1fr 1fr;gap:40px">'
        + card("Gold-set polarity, all 424 labeled pairs", "84% vs 64%", "text vs stars. Stars call 33% of these &#8220;neutral&#8221;.", inv=True, big_size=72)
        + card("Reviews with 2–4 stars (138 pairs)", "66% vs 0%", "Stars can&#8217;t say anything about a middle review. The text gets two-thirds right.", big_size=72)
        + '</div><p class="sm" style="margin-top:28px;font-size:26px;color:#111">The sentiment reader is a logistic regression on the review vectors, trained on 400,000 reviews. 74% accurate on 100,000 held-out reviews; its score correlates 0.80 with stars. Store scores from text and from stars rank stores alike (ρ 0.82–0.94).</p>')
slide(9, "Backup · Sentiment", "Why read good or bad from the text?", body, chip=prov())

body = table(["Outcome (statewide, Pipeline A)", "% Black", "Income ($10k)", "Interaction", "R²"],
             [["Distance to nearest store (z)", "−1.03***", "−0.03***", "0.03", "0.05"], ["Availability", "−0.36***", "0.00", "−0.05**", "0.02"],
              ["Accessibility", "−0.61***", "−0.01**", "−0.03*", "0.06"], ["Accommodation", "−0.37***", "0.00", "−0.05**", "0.02"],
              ["Affordability", "−0.24***", "−0.02***", "−0.03", "0.01"], ["Acceptability", "−0.71***", "−0.02***", "−0.08***", "0.07"]], cls="tb num", font=24)
slide(9, "Backup · Regression", "Statewide tract regression, Teresa&#8217;s method", body,
      "2,624 tracts, population-weighted, ACS 2022. * p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001.", chip=prov())

body = table(["Lesson", "What happened"],
             [["Verify by identity", "Name search sent the scraper to the wrong store. It now opens each store by Google ID and rejects a page whose ID differs."],
              ["A silent scroll looks like the end", "The scraper picked the wrong panel and declared stores complete after one screen. Every scroll now checks it moved."],
              ["Resumability is everything", "State lives in SQLite. An interrupted store restarts cleanly. Counts are re-read from disk, never kept in memory."],
              ["Two kinds of block", "A sign-in wall burns the account: rotate logins. A CAPTCHA burns the IP: pause 30–60 minutes."],
              ["Retry before you give up", "190 stores looked capped by Google. One more pass on a fresh session returned 24,481 more reviews."]], font=22)
slide(9, "Backup · Scraper", "Scraper engineering: what turned out to be hard", body)

body = table(["Term", "Meaning"],
             [["Topic", "A cluster of reviews about the same thing, found by the model"],
              ["Pooling", "Pulling small-sample store scores toward the average, more for stores with fewer reviews"],
              ["ρ (rho)", "Spearman rank correlation: 1 = same order, 0 = unrelated"],
              ["Recall / F1", "Share of true labels found / balance of finding them and being right"],
              ["Census tract", "A Census area of about 4,000 people; NC has 2,672 with data"],
              ["Frozen arm", "Teresa&#8217;s saved Raleigh model, applied unchanged to NC reviews"],
              ["Gold set", "The 260 reviews people label: the answer key"]], font=23)
slide(9, "Backup · Glossary", "Terms used in this deck", body)

SCRIPT = """<script>
const SECS=['Introduction','Goal','What we did','Results','Limits and next steps'];
class Deck{
  constructor(){
    this.slides=[...document.querySelectorAll('.slide')];this.stage=document.getElementById('deckStage');this.notes=document.getElementById('notes');this.buf='';
    const main=this.slides.filter(s=>s.dataset.sec!=='9').length;
    this.slides.forEach((s,i)=>{
      const sec=+s.dataset.sec;if(i===0)return;
      const ft=document.createElement('footer');ft.className='ft';
      const trk=sec>=1&&sec<=8?'<span class="trk">'+SECS.map((_,k)=>'<i class="'+(k+1<sec?'done':k+1===sec?'on':'')+'"></i>').join('')+'</span><span class="sec">'+String(sec).padStart(2,'0')+' · '+SECS[sec-1]+'</span>':sec===9?'<span class="sec">Backup</span>':'';
      const pg=i<main?String(i+1).padStart(2,'0')+' / '+main:'B'+(i-main+1);
      ft.innerHTML=trk+'<span class="who">NC Aspatial Food Access</span><span class="pg">'+pg+'</span>';
      s.appendChild(ft);});
    this.scale();addEventListener('resize',()=>this.scale());this.keys();
    this.show((parseInt(location.hash.slice(1),10)||1)-1);
  }
  scale(){const f=Math.min(innerWidth/1920,innerHeight/1080);this.stage.style.transform=`translate(${(innerWidth-1920*f)/2}px,${(innerHeight-1080*f)/2}px) scale(${f})`;}
  show(i){this.cur=Math.max(0,Math.min(i,this.slides.length-1));const s=this.slides[this.cur];
    this.slides.forEach((x,k)=>x.classList.toggle('active',k===this.cur));
    this.notes.textContent=s.dataset.notes||'';try{history.replaceState(null,'','#'+(this.cur+1));}catch(e){}}
  next(){this.show(this.cur+1)} prev(){this.show(this.cur-1)}
  keys(){document.addEventListener('keydown',e=>{if(e.metaKey||e.ctrlKey||e.altKey)return;const k=e.key;
    if(['ArrowRight','ArrowDown','PageDown',' '].includes(k)){e.preventDefault();this.next();}
    else if(['ArrowLeft','ArrowUp','PageUp','Backspace'].includes(k)){e.preventDefault();this.prev();}
    else if(k==='Home')this.show(0);else if(k==='End')this.show(this.slides.length-1);
    else if(/^[0-9]$/.test(k)){this.buf+=k;clearTimeout(this.bt);this.bt=setTimeout(()=>this.buf='',2500);}
    else if(k==='Enter'&&this.buf){this.show(parseInt(this.buf,10)-1);this.buf='';}
    else if(k==='f'||k==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen().catch(()=>{});}
    else if(k==='n'||k==='N')this.notes.classList.toggle('show');});
    addEventListener('click',e=>{if(!this.notes.contains(e.target))this.next();});}
}
const deck=new Deck();
</script>"""

html = ('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/><meta content="width=device-width, initial-scale=1.0" name="viewport"/>'
        f'<title>Food Access from Google Reviews · Suyash Sapre</title><style>{CSS}</style></head><body>'
        '<div class="deck-viewport"><main class="deck-stage" id="deckStage">' + "\n".join(SLIDES) +
        '</main></div><div class="notes" id="notes"></div>' + SCRIPT + "</body></html>")
(HERE / "Final_Presentation.html").write_text(html)
print(f"{len(SLIDES)} slides -> {HERE / 'Final_Presentation.html'}")
