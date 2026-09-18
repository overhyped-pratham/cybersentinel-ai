"""
CyberSentinel AI — Attack Story Engine Unit Tests.
"""

from pathlib import Path
import datetime
import pytest

from ml.defense.attack_story_engine import AttackStoryEngine, AttackStory, CorrelatedEventStep


@pytest.fixture
def engine():
    return AttackStoryEngine()


def test_empty_events_returns_baseline_story(engine):
    story = engine.build_story([])
    assert isinstance(story, AttackStory)
    assert story.severity == "LOW"
    assert story.peak_risk_score == 0.0
    assert story.event_count == 0
    assert "Baseline" in story.title


def test_emergent_attack_story_generation(engine):
    now = datetime.datetime.now(datetime.timezone.utc)
    t0 = now.isoformat()
    t1 = (now + datetime.timedelta(seconds=20)).isoformat()
    t2 = (now + datetime.timedelta(seconds=45)).isoformat()

    events = [
        {
            "event_id": "evt-1",
            "timestamp": t0,
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.10",
            "stage": "RECONNAISSANCE",
            "risk_score": 35.0,
            "anomaly_score": 0.2,
            "attack_probability": 0.8,
            "top_features": [{"feature": "dst_port_entropy", "attribution": 0.85}],
        },
        {
            "event_id": "evt-2",
            "timestamp": t1,
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.10",
            "stage": "CREDENTIAL_ACCESS",
            "risk_score": 65.0,
            "anomaly_score": 0.4,
            "attack_probability": 0.95,
            "top_features": [{"feature": "failed_flow_ratio", "attribution": 0.92}],
        },
        {
            "event_id": "evt-3",
            "timestamp": t2,
            "src_ip": "10.0.0.10",  # Pivoted from victim to source!
            "dst_ip": "10.0.0.25",
            "stage": "LATERAL_MOVEMENT",
            "risk_score": 88.0,
            "anomaly_score": 0.7,
            "attack_probability": 0.99,
            "top_features": [{"feature": "port_445_share", "attribution": 0.96}],
        },
    ]

    story = engine.build_story(events, story_id="story-test-01")
    assert story.story_id == "story-test-01"
    assert story.event_count == 3
    assert story.peak_risk_score == 88.0
    assert story.severity == "CRITICAL"
    assert story.tactical_progression == ["RECONNAISSANCE", "CREDENTIAL_ACCESS", "LATERAL_MOVEMENT"]
    
    # Step 3 should recognize pivoting (10.0.0.10 was dst in step 2, now src in step 3)
    step3 = story.steps[2]
    assert "Pivoting" in step3.link_rationale or step3.causal_confidence > 0.5
    
    # Check containment recommendations
    assert any("Isolate intermediate pivot" in r for r in story.containment_recommendations)
    assert any("credential reset" in r.lower() for r in story.containment_recommendations)


def test_story_retrieval(engine):
    events = [
        {
            "event_id": "e-1",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stage": "RECONNAISSANCE",
            "risk_score": 40.0,
        }
    ]
    story = engine.build_story(events, story_id="test-retrieve")
    retrieved = engine.get_story("test-retrieve")
    assert retrieved is not None
    assert retrieved.story_id == "test-retrieve"
    assert len(engine.list_stories()) >= 1
