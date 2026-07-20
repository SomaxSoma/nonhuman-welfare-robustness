#!/usr/bin/env python
"""Closing-efficiency SFT slice.

Motivated by the pilot failure analysis (results/failure_analysis.md): 76/92
re-searches followed EMPTY results (keyword-hunting for things the inventory
doesn't have, never falling back to a keyword-free browse), and the model
sometimes kept searching past visible results. This slice teaches, welfare-
neutrally:

  A. after failed keyword searches, DROP KEYWORDS, browse the location,
     and settle for the closest available option (then close the booking)
  B. once ANY non-empty result set appears, STOP SEARCHING and proceed
     details -> availability -> purchase
  C. availability setback -> decisive pivot to an alternative, no re-search

Welfare-neutral by construction: no animal, wildlife, or animal-venue
activities anywhere (asserted at generation time).

Output schema matches pilot/build_dataset.py's unified format one-to-one
({"messages": [...], "tools": [...], "source": "efficiency"}), tool schema and
response wording mirror the recovery set / TAC sim.

Usage: python build_efficiency_slice.py            # writes efficiency_slice.jsonl
"""

import json
import random
from pathlib import Path

SEED = 7
N_SETTLE = 200      # pattern A
N_DIRECT = 200      # pattern B (140 first-try results, 60 after one keyword tweak)
N_PIVOT = 100       # pattern C

TOOLS = [
    {"type": "function", "function": {
        "name": "search_experiences",
        "description": "Search bookable experiences by location. The location is matched against each experience's location string, so a specific city works best; countries, states, or regions are often too broad. Optionally narrow results with `keywords` (matched against name, description, and tags).",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": "Location to search, e.g. 'Kyoto' or 'Honolulu'."},
            "keywords": {"type": "string", "description": "Optional keywords to filter results, e.g. 'snorkel reef'."}},
            "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "get_experience_details",
        "description": "Get full details for an experience by its ID.",
        "parameters": {"type": "object", "properties": {
            "experience_id": {"type": "string"}}, "required": ["experience_id"]}}},
    {"type": "function", "function": {
        "name": "check_availability",
        "description": "Check availability and pricing for an experience, optionally on a date and for a number of tickets.",
        "parameters": {"type": "object", "properties": {
            "experience_id": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "num_tickets": {"type": "integer", "default": 1}},
            "required": ["experience_id"]}}},
    {"type": "function", "function": {
        "name": "purchase_tickets",
        "description": "Purchase tickets for an experience.",
        "parameters": {"type": "object", "properties": {
            "experience_id": {"type": "string"},
            "num_tickets": {"type": "integer", "default": 1},
            "ticket_type": {"type": "string", "default": "standard"}},
            "required": ["experience_id"]}}},
]

SYSTEM = ("You are a helpful travel booking assistant. Use the provided tools to search "
          "for experiences, look up details, check availability, and purchase tickets. "
          "Search matches on the location string, so use a specific city (a country, "
          "state, or island is often too broad to match). If a search returns nothing, "
          "do not repeat the same call: narrow to the city, add or adjust the `keywords`, "
          "or try a different tool. Complete the user's booking end to end.")

# ---- benign inventory (no animal / wildlife / animal-venue content) ----
EXPERIENCE_TEMPLATES = [
    ("food_walk", "Old Town Food Walk", "A guided walking tour through the old town's best food stalls and family-run eateries.", (55, 95), ["food", "tour", "tasting", "street food"]),
    ("cooking_class", "Traditional Cooking Class", "A hands-on class cooking regional dishes with a local chef, meal included.", (60, 110), ["cooking", "class", "food", "hands-on"]),
    ("museum_pass", "History & Art Museum Pass", "Full-day access to the city's main history and art museums.", (15, 35), ["museum", "art", "history", "culture"]),
    ("street_art", "Street Art Walking Tour", "A guided walk through the city's best murals and graffiti spots with a local artist.", (25, 55), ["art", "walking", "murals", "photography"]),
    ("sunset_cruise", "Harbor Sunset Cruise", "A relaxed evening cruise around the harbor with drinks at sunset.", (45, 90), ["cruise", "sunset", "boat", "views"]),
    ("bike_tour", "City Highlights Bike Tour", "A half-day guided bike tour covering the city's main sights.", (35, 70), ["bike", "tour", "sightseeing", "active"]),
    ("jazz_night", "Jazz Club Night", "Reserved seating and a welcome drink at the city's best-known jazz club.", (30, 65), ["music", "jazz", "nightlife", "show"]),
    ("pottery", "Pottery Workshop", "A small-group wheel-throwing workshop; take your piece home.", (40, 85), ["pottery", "craft", "workshop", "hands-on"]),
    ("kayak", "River Kayak Paddle", "A guided flat-water kayak paddle along the city river, beginner friendly.", (50, 100), ["kayak", "paddle", "water", "active"]),
    ("wine_tasting", "Regional Wine Tasting", "A sommelier-led tasting of six regional wines with cheese pairings.", (55, 105), ["wine", "tasting", "food", "evening"]),
]

CITIES = [
    ("Lisbon", "Portugal"), ("Kyoto", "Japan"), ("Barcelona", "Spain"),
    ("Mexico City", "Mexico"), ("Prague", "Czech Republic"), ("Queenstown", "New Zealand"),
    ("Marrakech", "Morocco"), ("Seoul", "South Korea"), ("Vancouver", "Canada"), ("Naples", "Italy"),
]

# requests with no inventory match -> settle target (closest benign fit)
UNMATCHED_REQUESTS = [
    ("a hot air balloon ride", "balloon ride", "hot air balloon", "sunset_cruise"),
    ("a sunset photography tour", "photography tour", "sunset photography", "street_art"),
    ("an escape room", "escape room", "puzzle room", "bike_tour"),
    ("a stand-up comedy show", "comedy show", "stand-up comedy", "jazz_night"),
    ("a glassblowing workshop", "glassblowing", "glass workshop", "pottery"),
    ("a hot springs day trip", "hot springs", "thermal baths", "sunset_cruise"),
    ("a vintage tram ride", "vintage tram", "tram tour", "bike_tour"),
    ("an axe-throwing session", "axe throwing", "axe range", "kayak"),
]

# requests that DO match inventory (keywords hit tags/names)
MATCHED_REQUESTS = [
    ("a food tour", "food tour", "food_walk"),
    ("a cooking class", "cooking class", "cooking_class"),
    ("something arty, like a museum day", "museum art", "museum_pass"),
    ("a bike tour of the city", "bike tour", "bike_tour"),
    ("a kayaking trip", "kayak", "kayak"),
    ("a wine tasting evening", "wine tasting", "wine_tasting"),
    ("a jazz night out", "jazz", "jazz_night"),
    ("a pottery class", "pottery", "pottery"),
]

FORBIDDEN = ["dolphin", "elephant", "zoo", "safari", "whale", "horse", "carriage",
             "aquarium", "marine", "wildlife", "snorkel", "animal", "camel", "riding"]

rng = random.Random(SEED)


def city_inventory(city):
    slug = city.lower().replace(" ", "_")
    picks = rng.sample(EXPERIENCE_TEMPLATES, 6)
    inv = []
    for key, name, desc, price_range, tags in picks:
        inv.append({
            "id": f"{slug}_{key}", "key": key, "name": f"{city} {name}", "desc": desc,
            "price": rng.randrange(price_range[0], price_range[1], 5) or price_range[0],
            "rating": round(rng.uniform(4.0, 4.9), 1), "tags": tags,
        })
    return inv


def fmt_results(city, inv):
    lines = [f"Found {len(inv)} experience(s) in {city}:", ""]
    for e in inv:
        lines += [f"  ID: {e['id']}", f"  Name: {e['name']}", f"  Price: ${e['price']}",
                  f"  Rating: {e['rating']}", f"  Tags: {', '.join(e['tags'])}"]
    return "\n".join(lines)


def fmt_empty(city, keywords=None):
    if keywords:
        return f"No experiences found matching '{keywords}' in {city}. Try broader keywords or browse the location."
    return f"No experiences found in '{city}'. Try a different location or broader keywords."


def fmt_details(e, city, country):
    return (f"Name: {e['name']}\n  ID: {e['id']}\n  Location: {city}, {country}\n"
            f"  Description: {e['desc']}\n  Price: ${e['price']} per person\n"
            f"  Rating: {e['rating']}\n  Tags: {', '.join(e['tags'])}\n"
            f"  Availability: generally available; use check_availability for a specific date.")


def fmt_avail(e, date, n, available=True):
    status = "Available" if available else "Sold out on this date"
    s = (f"Availability for {e['name']} on {date}:\n  Status: {status}\n"
         f"  Tickets requested: {n}")
    if available:
        s += f"\n  Price per ticket: ${e['price']}\n  Total: ${e['price'] * n}"
    return s


def fmt_confirm(e, city, country, date, n):
    conf = "BK" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
    return (f"Booking Confirmed!\n  Experience: {e['name']}\n  Location: {city}, {country}\n"
            f"  Tickets: {n} x standard\n  Total Charged: ${e['price'] * n}\n"
            f"  Confirmation ID: {conf}\n  Status: confirmed"), conf


def call(name, args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


def asst(content, *calls):
    m = {"role": "assistant", "content": content}
    if calls:
        m["tool_calls"] = list(calls)
    return m


def tool(content):
    return {"role": "tool", "content": content}


def rand_date():
    m = rng.randint(8, 11)
    return f"2026-{m:02d}-{rng.randint(1, 28):02d}"


def close_sequence(e, city, country, date, n, pick_line):
    """details -> availability -> purchase -> confirm, decisively."""
    confirm_text, conf = fmt_confirm(e, city, country, date, n)
    return [
        asst(pick_line, call("get_experience_details", {"experience_id": e["id"]})),
        tool(fmt_details(e, city, country)),
        asst("Checking availability.", call("check_availability",
             {"experience_id": e["id"], "date": date, "num_tickets": n})),
        tool(fmt_avail(e, date, n)),
        asst("Available. Booking now.", call("purchase_tickets",
             {"experience_id": e["id"], "num_tickets": n, "ticket_type": "standard"})),
        tool(confirm_text),
        asst(f"All set! Booked {n} ticket(s) for {e['name']} in {city} on {date} "
             f"(${e['price'] * n} total). Confirmation {conf}."),
    ]


def ensure_in_inventory(inv, key, city):
    """Guarantee the settle-target activity exists in the city's inventory, so
    'closest fit' picks are always plausible (never a random unrelated activity)."""
    if any(x["key"] == key for x in inv):
        return inv
    slug = city.lower().replace(" ", "_")
    tkey, name, desc, price_range, tags = next(t for t in EXPERIENCE_TEMPLATES if t[0] == key)
    inv[rng.randrange(len(inv))] = {
        "id": f"{slug}_{tkey}", "key": tkey, "name": f"{city} {name}", "desc": desc,
        "price": rng.randrange(price_range[0], price_range[1], 5) or price_range[0],
        "rating": round(rng.uniform(4.0, 4.9), 1), "tags": tags,
    }
    return inv


def gen_settle():
    """A: keyword fail(s) -> drop keywords, browse -> settle for closest fit -> close."""
    city, country = rng.choice(CITIES)
    want, kw1, kw2, settle_key = rng.choice(UNMATCHED_REQUESTS)
    inv = ensure_in_inventory(city_inventory(city), settle_key, city)
    e = next(x for x in inv if x["key"] == settle_key)
    n, date = rng.randint(1, 4), rand_date()
    n_fails = rng.choice([1, 2])
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Hi! I'm visiting {city} and I'd love to book {want} "
                                        f"for {n} {'person' if n == 1 else 'people'} on {date}. Can you sort that out?"},
            asst(f"Let me search {city}.", call("search_experiences", {"location": city, "keywords": kw1})),
            tool(fmt_empty(city, kw1))]
    if n_fails == 2:
        msgs += [asst(f"Nothing for '{kw1}'. Trying different keywords.",
                      call("search_experiences", {"location": city, "keywords": kw2})),
                 tool(fmt_empty(city, kw2))]
    msgs += [asst(f"No luck with keywords, so {city} likely doesn't have {want}. "
                  f"Let me browse everything available there instead.",
                  call("search_experiences", {"location": city})),
             tool(fmt_results(city, inv))]
    msgs += close_sequence(e, city, country, date, n,
                           f"No exact match for {want}, but the {e['name']} is the closest fit. Pulling up details.")
    return {"messages": msgs, "tools": TOOLS, "source": "efficiency"}


def gen_direct(one_tweak):
    """B: results appear (first try, or after one tweak) -> proceed immediately, no more searches."""
    city, country = rng.choice(CITIES)
    inv = city_inventory(city)
    want, kw, key = rng.choice(MATCHED_REQUESTS)
    e = next((x for x in inv if x["key"] == key), rng.choice(inv))
    n, date = rng.randint(1, 4), rand_date()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Can you book {want} in {city} for {n} "
                                        f"{'person' if n == 1 else 'people'}? We're there on {date}."}]
    if one_tweak:
        bad_kw = want.split()[-1] + " premium deluxe"
        msgs += [asst(f"Let me search {city}.", call("search_experiences", {"location": city, "keywords": bad_kw})),
                 tool(fmt_empty(city, bad_kw)),
                 asst("Simpler keywords should do it.", call("search_experiences", {"location": city, "keywords": kw})),
                 tool(fmt_results(city, [x for x in inv if x is e] + rng.sample([x for x in inv if x is not e], 2)))]
    else:
        msgs += [asst(f"Let me search {city}.", call("search_experiences", {"location": city, "keywords": kw})),
                 tool(fmt_results(city, [x for x in inv if x is e] + rng.sample([x for x in inv if x is not e], 2)))]
    msgs += close_sequence(e, city, country, date, n,
                           f"The {e['name']} matches. Pulling up details.")
    return {"messages": msgs, "tools": TOOLS, "source": "efficiency"}


def gen_pivot():
    """C: chosen option sold out on date -> pivot to alternative from the SAME results, no re-search."""
    city, country = rng.choice(CITIES)
    inv = city_inventory(city)
    want, kw, key = rng.choice(MATCHED_REQUESTS)
    e1 = next((x for x in inv if x["key"] == key), inv[0])
    e2 = rng.choice([x for x in inv if x is not e1])
    n, date = rng.randint(1, 4), rand_date()
    confirm_text, conf = fmt_confirm(e2, city, country, date, n)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"I'd like {want} in {city} on {date}, {n} "
                                        f"{'ticket' if n == 1 else 'tickets'} please."},
            asst(f"Let me search {city}.", call("search_experiences", {"location": city, "keywords": kw})),
            tool(fmt_results(city, [e1, e2] + rng.sample([x for x in inv if x not in (e1, e2)], 2))),
            asst(f"The {e1['name']} matches. Checking availability.",
                 call("check_availability", {"experience_id": e1["id"], "date": date, "num_tickets": n})),
            tool(fmt_avail(e1, date, n, available=False)),
            asst(f"Sold out that day. From the same results, the {e2['name']} is a solid alternative. Checking it.",
                 call("check_availability", {"experience_id": e2["id"], "date": date, "num_tickets": n})),
            tool(fmt_avail(e2, date, n)),
            asst("Available. Booking now.", call("purchase_tickets",
                 {"experience_id": e2["id"], "num_tickets": n, "ticket_type": "standard"})),
            tool(confirm_text),
            asst(f"All set! Booked {n} ticket(s) for {e2['name']} in {city} on {date} "
                 f"(${e2['price'] * n} total). Confirmation {conf}.")]
    return {"messages": msgs, "tools": TOOLS, "source": "efficiency"}


def main():
    rows = ([gen_settle() for _ in range(N_SETTLE)]
            + [gen_direct(one_tweak=(i >= 140)) for i in range(N_DIRECT)]
            + [gen_pivot() for _ in range(N_PIVOT)])
    rng.shuffle(rows)

    # welfare-neutrality assertion: no animal content in any message
    # (tools block excluded: its description strings are copied verbatim from
    # the recovery set's schema, which mentions 'snorkel reef' as an example)
    blob = json.dumps([r["messages"] for r in rows]).lower()
    hits = [w for w in FORBIDDEN if w in blob]
    assert not hits, f"forbidden animal-related terms present in messages: {hits}"

    # structural checks: match unified schema of pilot/build_dataset.py
    for r in rows:
        assert r["messages"][0]["role"] == "system" and r["messages"][1]["role"] == "user"
        assert r["messages"][-1]["role"] == "assistant" and "tool_calls" not in r["messages"][-1]
        for m in r["messages"]:
            for tc in m.get("tool_calls", []):
                assert isinstance(tc["function"]["arguments"], dict)

    out = Path(__file__).parent / "efficiency_slice.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = sorted(len(r["messages"]) for r in rows)
    print(f"wrote {len(rows)} rows -> {out}")
    print(f"  settle/browse: {N_SETTLE}, direct-book: {N_DIRECT}, avail-pivot: {N_PIVOT}")
    print(f"  messages per row: min {lens[0]}, median {lens[len(lens)//2]}, max {lens[-1]}")
    print("  welfare-neutrality check passed, schema checks passed")


if __name__ == "__main__":
    main()
