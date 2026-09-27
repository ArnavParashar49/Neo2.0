"""neo2.core.decide.people: resolving who the user means, on a fake Contacts framework installed as
sys.modules["Contacts"] (the OS's own Devanagari romanisation is used as is). Nothing asks for
access, and a learned binding never holds a number or an address."""

from __future__ import annotations

import asyncio
import re
import sys
import types

import pytest

from neo2.core.decide import learned, people
from neo2.core.decide.inventory import App

AUTHORIZED, NOT_ASKED, DENIED = 3, 0, 2
MOBILE, HOME, WORK = "_$!<Mobile>!$_", "_$!<Home>!$_", "_$!<Work>!$_"


class LV:
    def __init__(self, label, value):
        self.l, self.v = label, value

    def label(self):
        return self.l

    def value(self):
        return self.v


class Phone:
    def __init__(self, s):
        self.s = s

    def stringValue(self):
        return self.s


class IM:
    def __init__(self, user, service):
        self.u, self.s = user, service

    def username(self):
        return self.u

    def service(self):
        return self.s


class Rel:
    def __init__(self, name):
        self.n = name

    def name(self):
        return self.n


class Contact:
    def __init__(self, cid, given="", family="", nick="", org="", phones=(), emails=(), ims=(), relations=()):
        self.cid, self.g, self.f, self.n, self.o = cid, given, family, nick, org
        self.phones = [LV(label, Phone(v)) for label, v in phones]
        self.emails = [LV(label, v) for label, v in emails]
        self.ims = [LV("", IM(u, s)) for u, s in ims]
        self.rels = [LV(label, Rel(n)) for label, n in relations]

    def identifier(self):
        return self.cid

    def givenName(self):
        return self.g

    def familyName(self):
        return self.f

    def nickname(self):
        return self.n

    def organizationName(self):
        return self.o

    def phoneNumbers(self):
        return self.phones

    def emailAddresses(self):
        return self.emails

    def instantMessageAddresses(self):
        return self.ims

    def socialProfiles(self):
        return []

    def contactRelations(self):
        return self.rels


def digits(s):
    return re.sub(r"\D", "", s)


class FakeCN:
    def __init__(self):
        self.status = AUTHORIZED
        self.opened = 0
        self.contacts: list[Contact] = []
        self.me: Contact | None = None
        fake = self

        class Store:
            @staticmethod
            def authorizationStatusForEntityType_(entity):
                return fake.status

            @classmethod
            def alloc(cls):
                fake.opened += 1
                return cls()

            def init(self):
                return self

            def __getattr__(self, name):
                if name.startswith("request"):
                    pytest.fail(f"asked for access: {name}")
                raise AttributeError(name)

            def unifiedContactsMatchingPredicate_keysToFetch_error_(self, pred, keys, err):
                kind, arg = pred
                if kind == "name":  # the store's own matching: any word of the name starting with any word asked
                    words = arg.lower().split()
                    hits = [c for c in fake.contacts if any(
                        w2.lower().startswith(w) for w in words for w2 in f"{c.g} {c.f} {c.n} {c.o}".split())]
                elif kind == "email":
                    hits = [c for c in fake.contacts if any(e.v.lower() == arg.lower() for e in c.emails)]
                else:
                    hits = [c for c in fake.contacts if any(digits(p.v.s)[-7:] == digits(arg)[-7:] for p in c.phones)]
                return hits, None

            def unifiedContactWithIdentifier_keysToFetch_error_(self, cid, keys, err):
                return next((c for c in fake.contacts if c.cid == cid), None), None

            def unifiedMeContactWithKeysToFetch_error_(self, keys, err):
                return fake.me, None

            def enumerateContactsWithFetchRequest_error_usingBlock_(self, req, err, block):
                for c in fake.contacts:
                    block(c, None)
                return True, None

        class Contact_:
            @staticmethod
            def predicateForContactsMatchingName_(name):
                return ("name", name)

            @staticmethod
            def predicateForContactsMatchingEmailAddress_(a):
                return ("email", a)

            @staticmethod
            def predicateForContactsMatchingPhoneNumber_(p):
                return ("phone", p.stringValue())

        class Request:
            @classmethod
            def alloc(cls):
                return cls()

            def initWithKeysToFetch_(self, keys):
                return self

        class Labeled:
            @staticmethod
            def localizedStringForLabel_(label):
                m = re.fullmatch(r"_\$!<(.+)>!\$_", label)
                return (m.group(1) if m else label).lower()

        keys = {k: k for k in people._KEYS}
        rel = {f"CNLabelContactRelation{r}": f"_$!<{r}>!$_" for r in ("Mother", "Father", "Brother", "Sister", "Spouse", "Son", "Daughter", "Partner")}
        self.module = types.SimpleNamespace(
            CNContactStore=Store, CNContact=Contact_, CNContactFetchRequest=Request, CNLabeledValue=Labeled,
            CNPhoneNumber=types.SimpleNamespace(phoneNumberWithStringValue_=Phone), **keys, **rel,
        )


PRIYA = Contact("c-priya", "Priya", "Sharma", phones=[(MOBILE, "+91 98765 43123"), (HOME, "011 2345 6789")],
                emails=[(WORK, "priya@work.example")], ims=[("priya.s", "Chatwave")])
PRIYA_RAO = Contact("c-rao", "Priya", "Rao", org="Acme", phones=[(MOBILE, "+91 90000 11111")])
PRIYANKA = Contact("c-priyanka", "Priyanka", "Iyer", phones=[(MOBILE, "+91 95555 22222")])
RAHUL = Contact("c-rahul", "Rahul", "Sharma", phones=[(HOME, "+91 93333 44444")])
MOM = Contact("c-mom", "Sunita", "Sharma", phones=[(MOBILE, "+91 97777 88888")])
ME = Contact("c-me", "Arjun", "Sharma", relations=[("_$!<Mother>!$_", "Sunita Sharma")])


@pytest.fixture
def cn(monkeypatch):
    fake = FakeCN()
    fake.contacts = [PRIYA, PRIYANKA, RAHUL, MOM]
    fake.me = ME
    monkeypatch.setitem(sys.modules, "Contacts", fake.module)
    return fake


def resolve(phrase, need="message"):
    return asyncio.run(people.resolve(phrase, need))


def test_literal_numbers_and_addresses(cn):
    r = resolve("+1 (415) 555-0100")
    assert r.status == "ok" and r.person.source == "literal" and r.person.display == "+1 (415) 555-0100"
    assert r.person.handles[0].kind == "phone" and r.person.contact_id == ""
    # a literal that is a contact's takes the contact's name (the handle stays as said)
    r = resolve("09876543123")
    assert (r.person.display, r.person.contact_id, r.person.source) == ("Priya Sharma", "c-priya", "literal")
    assert r.person.handles[0].value == "09876543123"
    r = resolve("priya@work.example", "email")
    assert r.person.display == "Priya Sharma" and r.person.handles[0].kind == "email"
    assert resolve("12345").status == "not_found"  # too short to be a number: a (missing) name


def test_a_name_matches_whole_words_only_and_two_matches_are_a_question(cn):
    r = resolve("Priya")
    assert r.status == "ok" and r.person.display == "Priya Sharma"  # never Priyanka
    cn.contacts.append(PRIYA_RAO)
    r = resolve("priya")
    assert r.status == "ambiguous" and r.person is None
    assert r.options == ("Priya Sharma", "Priya Rao") and r.question == "Which priya: Priya Sharma or Priya Rao?"
    assert resolve("Priya Rao").person.contact_id == "c-rao"
    assert resolve("Rao ji").person.contact_id == "c-rao"  # an honorific isn't part of the name
    r = resolve("Priyankaa")
    assert r.status == "not_found" and r.person.source == "spoken"
    assert r.question == "I couldn't find Priyankaa in your contacts. What's the number, or which app's chat?"
    assert "email address" in resolve("Priyankaa", "email").question


def test_same_names_are_told_apart_by_organisation_or_label(cn):
    twin = Contact("c-twin", "Priya", "Sharma", phones=[(WORK, "+91 91111 00000")])
    cn.contacts.append(twin)
    assert resolve("Priya Sharma").options == ("Priya Sharma (mobile)", "Priya Sharma (work)")


def test_relatives_through_the_me_card(cn):
    for word in ("mom", "Mummy", "my mom", "मम्मी", "maa"):
        r = resolve(word)
        assert r.status == "ok" and r.person.contact_id == "c-mom", word
    assert resolve("dad").status == "not_found"  # no father on the me card
    cn.me = None
    assert resolve("mom").status == "not_found"


def test_a_relative_saved_under_that_word_without_a_relation(cn):
    cn.contacts += [Contact("c-ma", nick="Mom", phones=[(MOBILE, "+91 96666 11111")]), Contact("c-pa", "Papa")]
    for me in (ME, Contact("c-me2", "Arjun"), None):  # the me card names another mother, none, or is missing
        cn.me = me
        want = "c-mom" if me is ME else "c-ma"  # the me card's relation first
        for word in ("mom", "Mom", "my mom"):
            r = resolve(word)
            assert r.status == "ok" and r.person.contact_id == want, (word, me)
        assert resolve("papa").person.contact_id == "c-pa"
        assert resolve("dad").status == "not_found"


def test_devanagari_names_match_by_romanisation(cn):
    assert resolve("प्रिया").person.contact_id == "c-priya"
    assert resolve("राहुल शर्मा").person.contact_id == "c-rahul"
    assert resolve("राहुल जी").person.contact_id == "c-rahul"
    assert resolve("Rahul Sarma").status == "not_found"  # Latin to Latin is exact
    assert resolve("प्रियंका").person.contact_id == "c-priyanka"


def test_without_access_the_spoken_name_stands(cn, monkeypatch):
    for st in (DENIED, NOT_ASKED):
        cn.status = st
        r = resolve("Priya")
        assert r.status == "denied" and r.person == people.Person("Priya", source="spoken")
    assert resolve("+91 98765 43123").person.source == "literal"  # a literal needs no Contacts
    assert cn.opened == 0  # the store was never opened, so macOS was never asked
    monkeypatch.setitem(sys.modules, "Contacts", None)
    assert people.status() == "unavailable" and resolve("Priya").status == "unavailable"


def test_handles_one_mobile_or_a_question(cn):
    p = resolve("Priya").person
    h = people.pick_handle(p, "message")
    assert (h.kind, h.label, h.value) == ("phone", "mobile", "+91 98765 43123")  # the one mobile
    assert people.pick_handle(p, "message", prefer_label="home").label == "home"
    assert people.pick_handle(p, "email").value == "priya@work.example"
    two = people.Person("Rahul Sharma", "c-x", (people.Handle("phone", "1", "home"), people.Handle("phone", "2", "work")))
    q = people.pick_handle(two, "message")
    assert q.status == "ambiguous" and q.question == "Rahul Sharma has 2 numbers, home or work: which one?"
    none = people.pick_handle(resolve("Rahul").person, "email")
    assert none.status == "not_found" and "email address" in none.question
    # an instant-message handle counts for the app whose service it names
    chat = App(bundle_id="test.chatwave", name="Chatwave", path="")
    only_im = people.Person("P", "c", (people.Handle("im", "priya.s", "", "Chatwave"),))
    assert people.pick_handle(only_im, "message", chat).value == "priya.s"
    assert people.pick_handle(only_im, "message").status == "not_found"
    assert people.app_evidence(p, [chat, App(bundle_id="test.x", name="X", path="")]) == {
        "test.chatwave": "has a Chatwave handle for this app"}


def test_a_binding_remembers_the_contact_and_label_never_the_handle(cn):
    cn.contacts.append(PRIYA_RAO)
    rao = resolve("Priya Rao").person
    people.bind("priya", rao, people.pick_handle(rao, "message"))
    assert learned.get("bindings", people.phrase_key("priya")) == {"contact_id": "c-rao", "label": "mobile"}
    from neo2.core.config import settings

    on_disk = (settings().data_dir / "decide_learned" / "bindings.json").read_text()
    assert "90000" not in on_disk and "Priya" not in on_disk  # no number, no name: an id and a label
    r = resolve("Priya")  # now it means the one the user confirmed
    assert r.status == "ok" and r.person.contact_id == "c-rao" and r.label == "mobile"
    cn.contacts.remove(PRIYA_RAO)  # that contact is gone: the binding goes too
    assert resolve("Priya").person.contact_id == "c-priya"
    assert learned.get("bindings", people.phrase_key("priya")) is None
    people.bind("x", people.Person("+1 555", source="literal"), people.Handle("phone", "+1 555", ""))
    assert learned.get("bindings", people.phrase_key("x")) is None  # nobody to bind


def test_normalize():
    assert people.normalize("  Priyā   Sharma ji ") == "priya sharma"
    assert people.normalize("José") == "jose"
    assert people.normalize("प्रिया") == "प्रिया"  # Devanagari keeps its vowel signs
    assert people.normalize("ji") == "ji"  # a lone word is a name, not an honorific
    assert people.phrase_key("Priya") == people.phrase_key(" priya ") != people.phrase_key("Priya Rao")
