
"""Versioned evaluation dataset for the Multi-Channel RAG.

30 source-grounded cases:
- 8 PDF
- 8 SQL
- 8 Mixed
- 3 Multi-turn
- 3 Unanswerable

17 development and 13 held-out questions.

References were checked against the original Core Rules PDF
and the supplied SQLite snapshot.

This is a manually curated benchmark, not a RAGAS-generated
synthetic dataset.
"""

from collections import Counter

VERSION = "1.0"
CREATED = "2026-10-28"


def case(
    case_id,
    split,
    category,
    question,
    reference_answer,
    *,
    pdf=(),
    sql=None,
    seed=None,
    route=None,
    answerable=True,
    construction=None,
):
    """Create a benchmark case with traceable evidence."""

    if route is None and answerable:
        if pdf and sql:
            route = "mixed"
        elif pdf:
            route = "pdf"
        else:
            route = "sql"

    sql_evidence = None

    if sql is not None:
        sql_evidence = {
            "reference_sql": sql[0],
            "parameters": sql[1],
            "verified_rows": sql[2],
        }

    result = {
        "id": case_id,
        "split": split,
        "category": category,
        "construction": construction,
        "question": question,
        "reference_answer": reference_answer,
        "expected_route": route,
        "answerable": answerable,
        "pdf_evidence": [
            {"page": page, "anchor": anchor}
            for page, anchor in pdf
        ],
        "sql_evidence": sql_evidence,
    }

    if seed is not None:
        result["history_seed_question"] = seed

    return result


CASES = [

    # ======================================================
    # PDF-ONLY: GENERAL RULES
    # ======================================================

    case(
        "P01", "development", "pdf",
        "What is the rule for the GRENADE Stratagem: who can use it, at what distance, and how are mortal wounds determined?",
        "In your Shooting phase, select an eligible Grenades unit that has not shot and is not engaged; choose a visible enemy within 8 inches and not engaged with your units; roll six D6 and inflict one mortal wound per 4+. Cost 1CP under the Core Stratagems.",
        pdf=[(42, "TARGET: One Grenades unit")],
        construction="single_hop",
    ),

    case(
        "P02", "development", "pdf",
        "How are Stratagems paid for and limited per phase, and can one normally affect a Battle-shocked unit?",
        "Pay the listed CP cost; a Stratagem cannot be used more than once in the same phase. A player cannot normally use Stratagems to affect their Battle-shocked units, subject to special exceptions.",
        pdf=[
            (41, "reduce your CP total"),
            (11, "cannot use Stratagems to affect that unit"),
        ],
        construction="multi_hop",
    ),

    case(
        "P03", "development", "pdf",
        "In the Only War mission, how is an objective controlled and when/how many Victory points can be scored from objectives?",
        "Sum the OC of models within 3 inches horizontally and 5 inches vertically. Higher total controls the objective; ties are contested. Starting battle round 2, at the end of each player's Command phase, the active player gains 1VP per controlled objective, up to 3VP per turn.",
        pdf=[
            (58, "add together the OC characteristics"),
            (59, "Starting from the second battle round"),
        ],
        construction="multi_hop",
    ),

    case(
        "P04", "development", "pdf",
        "My unit Advanced this turn and has an Assault weapon. Can it shoot, which weapons can it use, and may it charge?",
        "The unit remains eligible to shoot but can resolve attacks only using Assault weapons. It cannot declare a charge that turn.",
        pdf=[
            (14, "A unit cannot shoot or declare a charge in the same turn that it Advanced"),
            (25, "only resolve attacks using Assault weapons"),
        ],
        construction="cross_section_exception",
    ),

    case(
        "P05", "held_out", "pdf",
        "May a Vehicle shoot while in Engagement Range of an enemy, and is there a hit-roll penalty?",
        "Yes. Monsters and Vehicles can shoot while engaged. Their ranged attacks normally suffer -1 to Hit, except attacks made with Pistols.",
        pdf=[
            (20, "Monster and Vehicle units are eligible to shoot"),
        ],
        construction="rule_exception",
    ),

    case(
        "P06", "held_out", "pdf",
        "If a Battle-shocked unit Falls Back, what tests are made, what happens on 1–2, and can it shoot later that turn?",
        "A Desperate Escape test is required for each model. Each result of 1 or 2 destroys one model. A unit that Fell Back cannot shoot or charge that turn.",
        pdf=[
            (11, "If it Falls Back, you must take a Desperate Escape test"),
            (14, "For each roll of 1-2"),
        ],
        construction="multi_hop",
    ),

    case(
        "P07", "development", "pdf",
        "Does a model with a 3+ Save gain the Benefit of Cover against an AP 0 ranged attack?",
        "No. A model with a Save characteristic of 3+ or better does not benefit from cover against AP 0 attacks.",
        pdf=[
            (44, "Models with a Save characteristic"),
        ],
        construction="rule_exception",
    ),

    case(
        "P08", "held_out", "pdf",
        "If passengers disembark after their Transport made a Normal move, can they move again, shoot, or charge?",
        "They count as having made a Normal move. They cannot move further that phase or charge that turn, but may otherwise act normally, including shooting when eligible.",
        pdf=[
            (17, "count as having made a Normal move themselves"),
        ],
        construction="rule_exception",
    ),

    # ======================================================
    # SQL-ONLY: DATABASE FACTS
    # ======================================================

    case(
        "S01", "development", "sql",
        "What is the Toughness of a Space Marines Intercessor Squad?",
        "The Space Marines Intercessor Squad has Toughness 4.",
        sql=(
            "SELECT d.name, m.t AS toughness FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name AND d.factionId = :faction",
            {"name": "Intercessor Squad", "faction": "SM"},
            [{"name": "Intercessor Squad", "toughness": "4"}],
        ),
        construction="sql_join",
    ),

    case(
        "S02", "held_out", "sql",
        "How many Wounds does a Brutalis Dreadnought have, and what is its armour Save?",
        "Brutalis Dreadnought: 12 Wounds and a 2+ Save.",
        sql=(
            "SELECT d.name, m.w AS wounds, m.sv AS save FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name",
            {"name": "Brutalis Dreadnought"},
            [{"name": "Brutalis Dreadnought", "wounds": "12", "save": "2+"}],
        ),
        construction="multi_attribute",
    ),

    case(
        "S03", "held_out", "sql",
        "Name the two ranged weapons listed for Necron Warriors and give their ranges.",
        "Gauss flayer: 24 inches. Gauss reaper: 12 inches.",
        sql=(
            "SELECT w.name, w.range FROM datasheets d JOIN datasheets_wargear w ON w.DatasheetId=d.Id WHERE d.name = :name AND w.type = 'Ranged' ORDER BY w.name",
            {"name": "Necron Warriors"},
            [
                {"name": "Gauss flayer", "range": "24"},
                {"name": "Gauss reaper", "range": "12"},
            ],
        ),
        construction="one_to_many",
    ),

    case(
        "S04", "development", "sql",
        "What is the Objective Control characteristic of an Intercessor Squad?",
        "An Intercessor Squad has Objective Control 2 per model.",
        sql=(
            "SELECT m.oc AS objective_control FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name AND d.factionId = :faction",
            {"name": "Intercessor Squad", "faction": "SM"},
            [{"objective_control": "2"}],
        ),
        construction="sql_join",
    ),

    case(
        "S05", "held_out", "sql",
        "For Ork Boyz, which model profiles are present, and what is the Toughness of each?",
        "Boyz include BOSS NOB and BOY profiles. Both have Toughness 5.",
        sql=(
            "SELECT m.name, m.t AS toughness FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name ORDER BY m.name",
            {"name": "Boyz"},
            [
                {"name": "BOSS NOB", "toughness": "5"},
                {"name": "BOY", "toughness": "5"},
            ],
        ),
        construction="one_to_many",
    ),

    case(
        "S06", "development", "sql",
        "What are the ranged and melee profiles of Abaddon's Talon of Horus, including range, attacks, Strength, AP and Damage?",
        "Talon of Horus ranged: range 24 inches, 4 attacks, Strength 5, AP -1, Damage 2. Melee: range Melee, 14 attacks, Strength 7, AP -3, Damage 1.",
        sql=(
            "SELECT w.name,w.type,w.range,w.a AS attacks,w.s AS strength,w.ap,w.d AS damage FROM datasheets d JOIN datasheets_wargear w ON w.DatasheetId=d.Id WHERE d.name = :name AND w.name = :weapon ORDER BY w.type",
            {"name": "Abaddon The Despoiler", "weapon": "Talon of Horus"},
            [
                {
                    "name": "Talon of Horus",
                    "type": "Melee",
                    "range": "Melee",
                    "attacks": "14",
                    "strength": "7",
                    "ap": "-3",
                    "damage": "1",
                },
                {
                    "name": "Talon of Horus",
                    "type": "Ranged",
                    "range": "24",
                    "attacks": "4",
                    "strength": "5",
                    "ap": "-1",
                    "damage": "2",
                },
            ],
        ),
        construction="multiple_weapon_profiles",
    ),

    case(
        "S07", "development", "sql",
        "Does Roboute Guilliman have the Monster keyword and the Character keyword?",
        "Yes. Roboute Guilliman has both Monster and Character keywords.",
        sql=(
            "SELECT k.keyword FROM datasheets d JOIN datasheets_keywords k ON k.DatasheetId=d.Id WHERE d.name = :name AND lower(k.keyword) IN ('monster','character') ORDER BY k.keyword",
            {"name": "Roboute Guilliman"},
            [{"keyword": "Character"}, {"keyword": "Monster"}],
        ),
        construction="keyword_relation",
    ),

    case(
        "S08", "development", "sql",
        "What are the Movement, Toughness and Wounds of the Astra Militarum Leman Russ Battle Tank?",
        "Leman Russ Battle Tank: Movement 10 inches, Toughness 11, Wounds 13.",
        sql=(
            "SELECT d.name,m.m AS movement,m.t AS toughness,m.w AS wounds FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name AND d.factionId = :faction",
            {"name": "Leman Russ Battle Tank", "faction": "AM"},
            [{
                "name": "Leman Russ Battle Tank",
                "movement": '10"',
                "toughness": "11",
                "wounds": "13",
            }],
        ),
        construction="faction_disambiguation",
    ),

    # ======================================================
    # MIXED: PDF + SQL REASONING
    # ======================================================

    case(
        "M01", "development", "mixed",
        "What Wound Roll does a Strength 10 weapon need against Abaddon The Despoiler?",
        "2+. Abaddon has Toughness 5, and Strength 10 is twice Toughness 5.",
        pdf=[(21, "Strength is TWICE (or more than twice)")],
        sql=(
            "SELECT m.t AS toughness FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name",
            {"name": "Abaddon The Despoiler"},
            [{"toughness": "5"}],
        ),
        construction="cross_source_reasoning",
    ),

    case(
        "M02", "development", "mixed",
        "What Wound Roll does a Strength 3 attack need against a Space Marines Intercessor Squad?",
        "5+. Intercessor Squad has Toughness 4, and Strength 3 is less than Toughness but more than half of it.",
        pdf=[(21, "Strength is LESS than the Toughness")],
        sql=(
            "SELECT m.t AS toughness FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name AND d.factionId = :faction",
            {"name": "Intercessor Squad", "faction": "SM"},
            [{"toughness": "4"}],
        ),
        construction="cross_source_boundary",
    ),

    case(
        "M03", "held_out", "mixed",
        "What Wound Roll does a Strength 5 attack need against a Brutalis Dreadnought?",
        "6+. Brutalis Dreadnought has Toughness 10, and Strength 5 is half of Toughness.",
        pdf=[(21, "Strength is HALF (or less than half)")],
        sql=(
            "SELECT m.t AS toughness FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name",
            {"name": "Brutalis Dreadnought"},
            [{"toughness": "10"}],
        ),
        construction="cross_source_boundary",
    ),

    case(
        "M04", "held_out", "mixed",
        "What Wound Roll does a Strength 7 attack need against Necron Warriors?",
        "3+. Necron Warriors have Toughness 4. Strength 7 is greater than 4 but less than twice 4.",
        pdf=[(21, "Strength is GREATER than the Toughness")],
        sql=(
            "SELECT m.t AS toughness FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name",
            {"name": "Necron Warriors"},
            [{"toughness": "4"}],
        ),
        construction="cross_source_boundary",
    ),

    case(
        "M05", "development", "mixed",
        "Three Space Marines Intercessor models are in range of an objective. If the enemy has total Objective Control 5, who controls it?",
        "The Intercessors control it. Their OC is 2 per model, giving 3 × 2 = 6, higher than the enemy's 5.",
        pdf=[(58, "add together the OC characteristics")],
        sql=(
            "SELECT m.oc AS objective_control FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name AND d.factionId = :faction",
            {"name": "Intercessor Squad", "faction": "SM"},
            [{"objective_control": "2"}],
        ),
        construction="cross_source_arithmetic",
    ),

    case(
        "M06", "development", "mixed",
        "An Intercessor Squad is Battle-shocked. Does its normal Objective Control of 2 per model still count when holding an objective?",
        "No. While Battle-shocked, its models have Objective Control 0 regardless of their normal OC 2.",
        pdf=[
            (11, "Objective Control characteristic of all of its models is 0"),
            (58, "add together the OC characteristics"),
        ],
        sql=(
            "SELECT m.oc AS normal_objective_control FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name AND d.factionId = :faction",
            {"name": "Intercessor Squad", "faction": "SM"},
            [{"normal_objective_control": "2"}],
        ),
        construction="cross_source_exception",
    ),

    case(
        "M07", "held_out", "mixed",
        "Can an Astra Militarum Leman Russ Battle Tank shoot its ranged weapons while within Engagement Range of an enemy, and what Hit-roll modifier applies?",
        "Yes. Leman Russ Battle Tank has the Vehicle keyword. Vehicles can shoot while engaged, normally with -1 to Hit except for Pistol attacks.",
        pdf=[(20, "Monster and Vehicle units are eligible to shoot")],
        sql=(
            "SELECT DISTINCT k.keyword FROM datasheets d JOIN datasheets_keywords k ON k.DatasheetId=d.Id WHERE d.name = :name AND d.factionId = :faction AND lower(k.keyword) = :keyword",
            {
                "name": "Leman Russ Battle Tank",
                "faction": "AM",
                "keyword": "vehicle",
            },
            [{"keyword": "Vehicle"}],
        ),
        construction="cross_source_type_exception",
    ),

    case(
        "M08", "held_out", "mixed",
        "Does a Space Marines Intercessor Squad have the keyword required for the GRENADE Stratagem, and how does that Stratagem deal mortal wounds?",
        "Yes. The Intercessor Squad has Grenades. If the other conditions are met, select a visible enemy within 8 inches and roll six D6, inflicting one mortal wound per result of 4+.",
        pdf=[(42, "TARGET: One Grenades unit")],
        sql=(
            "SELECT k.keyword FROM datasheets d JOIN datasheets_keywords k ON k.DatasheetId=d.Id WHERE d.name = :name AND d.factionId = :faction AND lower(k.keyword) = :keyword",
            {
                "name": "Intercessor Squad",
                "faction": "SM",
                "keyword": "grenades",
            },
            [{"keyword": "Grenades"}],
        ),
        construction="cross_source_keyword_rule",
    ),

    # ======================================================
    # MULTI-TURN: CONVERSATION HISTORY
    # ======================================================

    case(
        "T01", "development", "multiturn",
        "What weapons does he have?",
        "Abaddon The Despoiler has Drach'nyen (melee) and Talon of Horus (ranged and melee).",
        sql=(
            "SELECT w.name,w.type FROM datasheets d JOIN datasheets_wargear w ON w.DatasheetId=d.Id WHERE d.name = :name ORDER BY w.name,w.type",
            {"name": "Abaddon The Despoiler"},
            [
                {"name": "Drachâ\x80\x99nyen", "type": "Melee"},
                {"name": "Talon of Horus", "type": "Melee"},
                {"name": "Talon of Horus", "type": "Ranged"},
            ],
        ),
        seed="What is Abaddon's Toughness?",
        route="sql",
        construction="conversation_reference",
    ),

    case(
        "T02", "held_out", "multiturn",
        "What would a Strength 8 weapon need to wound it?",
        "2+. Intercessor Squad has Toughness 4, and Strength 8 is twice its Toughness.",
        pdf=[(21, "Strength is TWICE (or more than twice)")],
        sql=(
            "SELECT m.t AS toughness FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name AND d.factionId = :faction",
            {"name": "Intercessor Squad", "faction": "SM"},
            [{"toughness": "4"}],
        ),
        seed="What is the Toughness of a Space Marines Intercessor Squad?",
        construction="conversation_reference_cross_source",
    ),

    case(
        "T03", "development", "multiturn",
        "If four of them are within range of an objective and the opponent has total OC 7, who controls it?",
        "The four Necron Warriors control it. Their OC is 2 each, giving 8, higher than the opponent's 7.",
        pdf=[(58, "add together the OC characteristics")],
        sql=(
            "SELECT m.oc AS objective_control FROM datasheets d JOIN datasheets_models m ON m.DatasheetId=d.Id WHERE d.name = :name",
            {"name": "Necron Warriors"},
            [{"objective_control": "2"}],
        ),
        seed="What is the Objective Control characteristic of Necron Warriors?",
        construction="conversation_reference_cross_source",
    ),

    # ======================================================
    # UNANSWERABLE: NEGATIVE CONTROLS
    # ======================================================

    case(
        "U01", "development", "unanswerable",
        "What is the Toughness of the Azure Void Dragonlets unit?",
        "The provided database contains no datasheet for Azure Void Dragonlets. Its Toughness cannot be determined.",
        answerable=False,
        route="sql",
        construction="missing_entity",
    ),

    case(
        "U02", "held_out", "unanswerable",
        "What is the current retail price in euros of an Intercessor Squad box?",
        "Not answerable from the available sources, which do not contain current retail prices.",
        answerable=False,
        construction="outside_corpus",
    ),

    case(
        "U03", "held_out", "unanswerable",
        "How many Warhammer 40,000 matches have I personally won this year?",
        "Not answerable from the provided sources, which contain no personal match history.",
        answerable=False,
        construction="missing_personal_context",
    ),
]


def validate_structure():
    """Validate benchmark metadata and structure offline."""

    assert len(CASES) == 30, "Expected 30 cases"

    ids = [c["id"] for c in CASES]
    assert len(set(ids)) == 30, "Duplicate case IDs"

    splits = Counter(c["split"] for c in CASES)
    categories = Counter(c["category"] for c in CASES)

    assert splits == {
        "development": 17,
        "held_out": 13,
    }, splits

    assert categories == {
        "pdf": 8,
        "sql": 8,
        "mixed": 8,
        "multiturn": 3,
        "unanswerable": 3,
    }, categories

    for c in CASES:
        assert c["question"]
        assert c["reference_answer"]
        assert c["expected_route"] in {
            "pdf", "sql", "mixed", None
        }

    return splits, categories


if __name__ == "__main__":
    splits, categories = validate_structure()

    print("Benchmark version:", VERSION)
    print("Total questions:", len(CASES))
    print("Splits:", dict(splits))
    print("Categories:", dict(categories))
    print("Validation: PASS")
    print("Gemini calls: 0")
