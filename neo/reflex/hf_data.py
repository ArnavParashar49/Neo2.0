"""Real human phrasings for the reflex, from open Hugging Face intent datasets.

Our own training data is templates plus LLM-generated sentences — clean, but nobody talks like
that. CLINC150 and MASSIVE are ~27k crowd-sourced assistant utterances ("you're too loud",
"emails", "olly make me a coffee", "how bad is traffic on my commute"); their intents map onto
NEO's routes (chat / quick_action / agent_task / stop) and, where NEO has a matching tool,
onto that tool. The mapping is the whole point: we don't learn *their* labels, we learn how
people phrase *ours*.

    python -m neo.reflex.hf_data            # writes ~/.neo/reflex_hf.jsonl (cached datasets)

Rows: {"text", "intent", "tool", "source"}; `tool` is "" when no single NEO tool applies.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

from neo.config import settings

# (intent, tool). A tool means "this one tool, called once, does it".
CLINC: dict[str, tuple[str, str]] = {
    # ---- conversation / knowledge → chat
    **dict.fromkeys(
        """restaurant_reviews nutrition_info oil_change_how gas_type meaning_of_life
        improve_credit_score measurement_conversion flip_coin do_you_have_pets tell_joke
        exchange_rate meal_suggestion tire_change user_name what_are_your_hobbies jump_start
        vaccines food_last who_do_you_work_for international_visa translate carry_on insurance
        what_is_your_name where_are_you_from cook_time roll_dice who_made_you how_old_are_you
        recipe yes no maybe ingredients_list what_can_i_ask_you are_you_a_bot plug_type
        oil_change_when thank_you fun_fact ingredient_substitution calories calculator definition
        next_holiday mpg spelling greeting goodbye repeat change_ai_name change_user_name
        change_accent change_speed whisper_mode change_language timezone interest_rate
        redeem_rewards oos reset_settings sync_device gas tire_pressure last_maintenance
        international_fees apr min_payment
       """.split(),
        ("chat", ""),
    ),
    # ---- needs current information from the web → quick_action + web_search
    **dict.fromkeys(
        """weather directions distance traffic direct_deposit flight_status travel_alert
        travel_suggestion restaurant_suggestion how_busy exchange_rate""".split(),
        ("quick_action", "web_search"),
    ),
    # ---- one-shot commands → quick_action (+ the NEO tool when there is one)
    "time": ("quick_action", "clock"),
    "date": ("quick_action", "clock"),
    "timer": ("quick_action", "timer"),
    "alarm": ("quick_action", "reminder_add"),
    "reminder_update": ("quick_action", "reminder_add"),
    "calendar": ("quick_action", "calendar_today"),
    "meeting_schedule": ("quick_action", "calendar_today"),
    "calendar_update": ("quick_action", "calendar_add"),
    "change_volume": ("quick_action", "volume"),
    "play_music": ("quick_action", ""),
    "next_song": ("quick_action", ""),
    "what_song": ("quick_action", ""),
    "smart_home": ("quick_action", ""),
    # ---- needs an app, a website, an account or several steps → agent_task
    **dict.fromkeys(
        """account_blocked accept_reservations report_lost_card order schedule_meeting
        freeze_account restaurant_reservation make_call text bill_balance balance uber car_rental
        credit_limit shopping_list expiration_date routing todo_list card_declined
        rewards_balance share_location book_flight insurance_change todo_list_update
        cancel_reservation transactions credit_score report_fraud spending_history reminder
        payday find_phone order_status confirm_reservation damaged_card pin_change
        replacement_card_duration new_card income taxes pto_request rollover_401k
        pto_request_status application_status update_playlist credit_limit_change pay_bill
        lost_luggage book_hotel w2 shopping_list_update pto_balance order_checks
        schedule_maintenance transfer current_location travel_notification pto_used bill_due""".split(),
        ("agent_task", ""),
    ),
    "cancel": ("stop", ""),
}

MASSIVE: dict[str, tuple[str, str]] = {
    "alarm_set": ("quick_action", "reminder_add"),
    "alarm_query": ("agent_task", ""),
    "alarm_remove": ("agent_task", ""),
    "audio_volume_down": ("quick_action", "volume"),
    "audio_volume_up": ("quick_action", "volume"),
    "audio_volume_mute": ("quick_action", "volume"),
    "audio_volume_other": ("quick_action", "volume"),
    "calendar_query": ("quick_action", "calendar_today"),
    "calendar_set": ("quick_action", "calendar_add"),
    "calendar_remove": ("agent_task", ""),
    "cooking_query": ("chat", ""),
    "cooking_recipe": ("chat", ""),
    "datetime_convert": ("chat", ""),
    "datetime_query": ("quick_action", "clock"),
    "email_addcontact": ("agent_task", ""),
    "email_query": ("quick_action", "mail_unread"),
    "email_querycontact": ("agent_task", ""),
    "email_sendemail": ("agent_task", "mail_send"),
    "general_greet": ("chat", ""),
    "general_joke": ("chat", ""),
    "general_quirky": ("chat", ""),
    **dict.fromkeys(
        """iot_cleaning iot_coffee iot_hue_lightchange iot_hue_lightdim iot_hue_lightoff
        iot_hue_lighton iot_hue_lightup iot_wemo_off iot_wemo_on""".split(),
        ("quick_action", ""),
    ),
    "lists_createoradd": ("agent_task", ""),
    "lists_query": ("agent_task", ""),
    "lists_remove": ("agent_task", ""),
    "music_dislikeness": ("chat", ""),
    "music_likeness": ("chat", ""),
    "music_query": ("chat", ""),
    "music_settings": ("quick_action", ""),
    "news_query": ("quick_action", "web_search"),
    "play_audiobook": ("quick_action", ""),
    "play_game": ("quick_action", ""),
    "play_music": ("quick_action", ""),
    "play_podcasts": ("quick_action", ""),
    "play_radio": ("quick_action", ""),
    "qa_currency": ("quick_action", "web_search"),  # rates change: needs the web
    "qa_definition": ("chat", ""),
    "qa_factoid": ("chat", ""),
    "qa_maths": ("chat", ""),
    "qa_stock": ("quick_action", "web_search"),
    "recommendation_events": ("quick_action", "web_search"),
    "recommendation_locations": ("quick_action", "web_search"),
    "recommendation_movies": ("quick_action", "web_search"),
    "social_post": ("agent_task", ""),
    "social_query": ("agent_task", ""),
    "takeaway_order": ("agent_task", ""),
    "takeaway_query": ("agent_task", ""),
    "transport_query": ("quick_action", "web_search"),
    "transport_taxi": ("agent_task", ""),
    "transport_ticket": ("agent_task", ""),
    "transport_traffic": ("quick_action", "web_search"),
    "weather_query": ("quick_action", "web_search"),
}

_WAKE_WORDS = ("olly ", "alexa ", "hey computer, ", "computer, ")


def _clean(text: str) -> str:
    t = " ".join(text.strip().split())
    low = t.lower()
    for w in _WAKE_WORDS:  # MASSIVE addresses "olly"; NEO's wake word is stripped before the reflex
        if low.startswith(w):
            t = t[len(w) :]
            break
    return t.strip(" ,.")


def load_rows(per_label: int = 60, seed: int = 0) -> list[dict]:
    """Balanced sample per source intent (60 × 210 intents ≈ 10k rows), mapped to NEO labels."""
    from datasets import load_dataset

    rnd = random.Random(seed)
    rows: list[dict] = []
    unmapped: Counter = Counter()

    clinc = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = clinc.features["intent"].names
    by: dict[str, list[str]] = {}
    for r in clinc:
        by.setdefault(names[r["intent"]], []).append(r["text"])
    for label, texts in by.items():
        if label not in CLINC:
            unmapped[f"clinc:{label}"] += len(texts)
            continue
        intent, tool = CLINC[label]
        for t in rnd.sample(texts, min(per_label, len(texts))):
            rows.append({"text": _clean(t), "intent": intent, "tool": tool, "source": f"clinc:{label}"})

    massive = load_dataset("mteb/amazon_massive_intent", "en", split="train")
    by = {}
    for r in massive:
        by.setdefault(r["label"], []).append(r["text"])
    for label, texts in by.items():
        if label not in MASSIVE:
            unmapped[f"massive:{label}"] += len(texts)
            continue
        intent, tool = MASSIVE[label]
        for t in rnd.sample(texts, min(per_label, len(texts))):
            t = _clean(t)
            if len(t.split()) >= 2 or intent == "stop":  # "emails" alone is too ambiguous to learn from
                rows.append({"text": t, "intent": intent, "tool": tool, "source": f"massive:{label}"})

    if unmapped:
        print(f"unmapped intents (skipped): {dict(unmapped)}", file=sys.stderr)
    seen: set[str] = set()
    out = []
    for r in rows:
        k = r["text"].lower()
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    rnd.shuffle(out)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=60)
    ap.add_argument("--out", type=Path, default=settings().data_dir / "reflex_hf.jsonl")
    args = ap.parse_args()
    rows = load_rows(args.per_label)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_intent = Counter(r["intent"] for r in rows)
    by_tool = Counter(r["tool"] for r in rows if r["tool"])
    print(f"{len(rows)} rows → {args.out}")
    print("intents:", dict(by_intent))
    print("tools:", dict(by_tool))


if __name__ == "__main__":
    main()
