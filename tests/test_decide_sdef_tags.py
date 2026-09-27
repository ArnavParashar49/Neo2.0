"""neo2.core.decide.sdef: the read tags and property metadata, from small dictionaries written here
(the shapes the survey found in every local dictionary). Nothing is launched or scripted."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from neo2.core.decide import sdef


def parse(body: str) -> sdef.Dictionary:
    return sdef.parse(ET.fromstring(f'<dictionary><suite name="S" code="SSSS">{body}</suite></dictionary>'))


APP = '<class name="application" code="capp">{}</class>'

NOTES = APP.format('<element type="note"/>') + """
<class name="note" code="note">
  <property name="name" code="pnam" type="text"/>
  <property name="body" code="body" type="text" description="the HTML content of the note"/>
  <property name="plaintext" code="pltx" type="text" access="r" description="the plaintext content"/>
  <property name="id" code="ID  " type="text" access="r"/>
</class>"""

REMINDERS = APP.format('<element type="list"/>') + """
<class name="list" code="list"><element type="reminder"/></class>
<class name="reminder" code="remi">
  <property name="name" code="pnam" type="text"/>
  <property name="completed" code="comb" type="boolean"/>
  <property name="creation date" code="ascd" type="date" access="r"/>
</class>"""

CALENDAR = APP.format('<element type="calendar"/>') + """
<class name="calendar" code="wres"><element type="event"/><property name="writable" code="wr05" type="boolean" access="r"/></class>
<class name="event" code="wrev"><property name="summary" code="wr11" type="text"/><property name="start date" code="wr1s" type="date"/></class>"""

MAIL = APP.format('<element type="account"/><property name="inbox" code="inmb" type="mailbox" access="r"/>') + """
<class name="account" code="mact"><element type="mailbox"/></class>
<class name="mailbox" code="mbxp"><element type="message"/><property name="unread count" code="mbuc" type="integer" access="r"/></class>
<class name="message" code="mssg"><property name="read status" code="isrd" type="boolean"/><property name="subject" code="subj" type="text" access="r"/></class>"""

BROWSER = """<command name="search the web" code="sfrisrch">
  <parameter name="in" code="dcnm" type="document" optional="yes"/><parameter name="for" code="qury" type="text"/></command>"""
PLAYER = """<command name="search" code="hookSrch" description="search a playlist">
  <direct-parameter type="playlist"/><parameter name="for" code="pTrm" type="text"/></command>"""


def test_read_tags_from_each_kind_of_dictionary():
    assert sdef.read_caps(parse(NOTES)) == {"note.read"}
    assert sdef.read_caps(parse(REMINDERS)) == {"reminder.read"}
    assert sdef.read_caps(parse(CALENDAR)) == {"event.read"}
    assert sdef.read_caps(parse(MAIL)) == {"mail.read", "mail.unread_count"}
    assert sdef.read_caps(parse(BROWSER)) == {"web.search"}
    assert sdef.read_caps(None) == frozenset()
    # the doing-tags are unchanged by any of this
    assert sdef.caps(parse(NOTES)) == frozenset() and "note.read" not in sdef.ALL_CAPS


def test_read_tags_need_the_real_thing():
    # a player's search scoped to a playlist (or a photo library's) is not a search for text as such
    assert "web.search" not in sdef.read_caps(parse(PLAYER))
    # an event class nothing contains can't be listed
    lone = CALENDAR.replace('<element type="event"/>', "")
    assert "event.read" not in sdef.read_caps(parse(lone))
    # a to-do list without a completed flag, a message class without a read status, a write-only body
    assert sdef.read_caps(parse(REMINDERS.replace("completed", "flagged").replace("comb", "flgd"))) == frozenset()
    assert "mail.read" not in sdef.read_caps(parse(MAIL.replace("isrd", "xxxx").replace("read status", "junk")))
    write_only = NOTES.replace('code="body" type="text"', 'code="body" type="text" access="w"').replace(
        '<property name="plaintext" code="pltx" type="text" access="r" description="the plaintext content"/>', ""
    )
    assert sdef.read_caps(parse(write_only)) == frozenset()


def test_read_tags_by_codes_alone():
    d = parse(APP.format('<element type="thing"/>') + """
      <class name="thing" code="wrev"><property name="starts" code="wr1s" type="date"/></class>
      <class name="letter" code="mssg"><property name="seen" code="isrd" type="boolean"/></class>""")
    assert sdef.read_caps(d) == {"event.read", "mail.read"}


def test_property_access_and_description():
    d = parse(NOTES)
    assert d.prop_access("note", "body") == "rw" and d.prop_access("note", "plaintext") == "r"
    assert d.prop_access("note", "id") == "r" and d.prop_access("note", "missing") == ""
    assert "HTML" in d.prop_desc("note", "body") and d.prop_desc("note", "name") == ""
    # inherited through the class chain, like the properties themselves
    sub = parse(NOTES + '<class name="quick note" code="qnot" inherits="note"/>')
    assert sub.prop_access("quick note", "plaintext") == "r" and "HTML" in sub.prop_desc("quick note", "body")
    # a dictionary built without metadata (older code, tests): readable and writable by default
    bare = sdef.Dictionary(commands={}, classes={"note": ("note", ("body",))}, properties={"body": "body"}, enums={},
                           class_props={"note": {"body": ("body", "text")}})
    assert bare.prop_access("note", "body") == "rw"


def test_prop_meta_leaves_the_hash_alone():
    a, b = parse(NOTES), parse(NOTES.replace("the HTML content", "the rich content"))
    assert hash(a) == hash(b) and a.prop_meta != b.prop_meta
