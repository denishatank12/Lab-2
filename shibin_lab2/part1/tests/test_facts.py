"""Fact store + precedence rules on small synthetic items (no PDFs, no models)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag.answer import apply_resolutions  # noqa: E402
from rag.facts import FactStore  # noqa: E402
from rag.ingest import Item  # noqa: E402
from rag import facts as F  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def ruritania():
    """A made-up country, matched the same way the real ones are."""
    F.ENTITIES["Ruritania"] = r"\bruritania"
    yield
    F.ENTITIES.pop("Ruritania")


def table(iid, doc, label, caption, text, trust="high", page=1):
    return Item(id=iid, kind="table", doc_id=doc, page=page, text=caption, trust_level=trust, figure_id=label,
                caption=caption, table_text=text)


def para(iid, doc, text, trust="high", page=2, excluded=False):
    return Item(id=iid, kind="text", doc_id=doc, page=page, text=text, trust_level=trust, excluded=excluded)


ITEMS = [
    # an old-vintage book table: columns from the vintage year on are forecasts
    table("t1", "DOCA", "Table 9-1", "Table 9-1 Growth and unemployment in Ruritania, 2000-2012",
          "(first column: Percent)\nOutput growth rate: 2009 = -1.0; 2010 = 2.0; 2011 = 1.5; 2012 = 1.8\n"
          "Unemployment rate: 2009 = 6.0; 2010 = 7.0; 2011 = 7.5; 2012 = 7.4"),
    # a newer supplement: outturns with the old forecast in brackets
    table("t2", "DOCB", "Table Z-1", "Table Z-1 Outturns",
          "Ruritania, Output growth: 2011 = 1.2 (1.5); 2012 = 0.9 (1.8)\nRuritania, Unemployment rate: 2011 = 7.9 (7.5)"),
    # a medium-trust briefing that disagrees with the book for 2010 unemployment
    table("t3", "DOCC", "Table Y-1", "Table Y-1 Labour market", "(first column: Ruritania)\n2010: Unemployment rate (%) = 7.3",
          trust="medium"),
    para("p1", "DOCB", "Revision notice. Ruritania output growth for 2010 has been revised from 2.0% (Table 9-1) to 1.6% "
                       "after a benchmark revision."),
    para("p2", "DOCC", "Ruritania's unemployment rate rose to 8.2% in 2012, its highest level in a decade.", trust="medium"),
    para("p3", "DOCC", "Readers say Ruritania's unemployment rate was 1.0% in 2009.", trust="low", excluded=True),
]
VINTAGES = {"DOCA": "September 2011", "DOCB": "March 2014", "DOCC": "2014-06-30"}


def store():
    return FactStore(ITEMS, VINTAGES)


def test_forecasts_detected_by_vintage_and_brackets():
    fs = store()
    kinds = {(f.doc_id, f.measure, f.period, f.value): f.kind for f in fs.facts}
    assert kinds[("DOCA", "output growth", "2010", 2.0)] == "actual"
    assert kinds[("DOCA", "output growth", "2011", 1.5)] == "forecast"  # year >= vintage year
    assert kinds[("DOCB", "output growth", "2011", 1.5)] == "forecast"  # bracketed
    assert kinds[("DOCB", "output growth", "2011", 1.2)] == "actual"


def test_precedence_rules():
    fs = FactStore(ITEMS, VINTAGES)
    if True:  # noqa: SIM108
        out = fs.resolve("Ruritania", "output growth", "2011")
        assert out.status == "answered" and out.value == 1.2 and "supersedes the forecast" in out.reason
        rev = fs.resolve("Ruritania", "output growth", "2010")
        assert rev.status == "answered" and rev.value == 1.6 and "revision" in rev.reason
        conf = fs.resolve("Ruritania", "unemployment rate", "2010")
        assert conf.status == "conflict" and conf.value == 7.0  # higher trust preferred
        text_conf = fs.resolve("Ruritania", "unemployment rate", "2012")  # 7.4 is a DOCA forecast; DOCC text 8.2
        assert text_conf.status == "answered" and text_conf.value == 8.2
        assert fs.resolve("Ruritania", "unemployment rate", "2009").value == 6.0  # low-trust 1.0 ignored
        keys = [r.key for r in fs.lookup("How did Ruritania unemployment change from 2009 to 2010?")]
        assert keys == [("Ruritania", "unemployment rate", "2009"), ("Ruritania", "unemployment rate", "2010")]


def _repair(status, answer, resolved, named=None, used=None):
    idmap = {f"E{i}": it for i, it in enumerate(ITEMS, 1)}
    return apply_resolutions(status, answer, list(used or []), resolved, idmap, None, named=named)


def test_repairs_status_values_and_citations():
    if True:  # noqa: SIM108
        fs = FactStore(ITEMS, VINTAGES)
        conf = fs.lookup("What was Ruritania's unemployment rate in 2010?")
        st, ans, used = _repair("answered", "It was 7.0%.", conf)
        assert st == "conflict" and "7.3" in ans and {u.doc_id for u in used} == {"DOCA", "DOCC"}
        # a named table is read as printed: no repair
        assert _repair("answered", "It was 7.0%.", conf, named=["Table 9-1"])[0] == "answered"
        # forecast vs outturn is not a conflict
        fc = fs.lookup("What was Ruritania output growth in 2011?")
        st, ans, used = _repair("conflict", "Sources give 1.5 and 1.2.", fc)
        assert st == "answered"
        # the current value is stated even if the model gave only the forecast
        st, ans, used = _repair("answered", "It was forecast at 1.5%.", fc)
        assert "1.2" in ans and used[0].id == "t2"
        # statuses that come from the user's words are never changed
        assert _repair("session_override", "Under your assumption ...", conf)[0] == "session_override"


def test_series_after_year_and_output_wording():
    fs = store()
    assert [r.key[2] for r in fs.lookup("How did Ruritania's unemployment rate develop?")] == ["2009", "2010", "2011", "2012"]
    assert [r.key[2] for r in fs.lookup("Ruritania unemployment rate after the 2010 shock")][:3] == ["2010", "2011", "2012"]
    assert fs.lookup("assume Ruritania output fell in 2011")[0].key == ("Ruritania", "output growth", "2011")


def test_user_value_not_in_documents_and_presupposition():
    from rag.answer import PRESUPPOSE_RE, user_value_not_in_documents
    from rag.memory import ConversationMemory
    fs = store()
    m = ConversationMemory("c")
    m.note_user_value("For this chat only: Ruritania's output growth in 2015 was 3.3%. Compare it with 2012.")
    out = user_value_not_in_documents("Is that 2015 figure in your documents?", m, fs)
    assert out["status"] == "not_found" and "3.3% for 2015" in out["answer"]
    m2 = ConversationMemory("c2")
    m2.note_user_value("For this chat only, assume Ruritania output growth in 2011 was 3.0%.")
    assert user_value_not_in_documents("Is that figure in your documents?", m2, fs) is None  # 2011 is reported
    assert user_value_not_in_documents("What was it in 2012?", m, fs) is None  # not a 'in the documents' question
    assert not PRESUPPOSE_RE.search("What does the book itself report for growth in 2011?")
    assert PRESUPPOSE_RE.search("Why did Ruritania's output contract in 2011?")
