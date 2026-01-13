"""
Pytest fixtures and configuration for Wenah tests.
"""

import pytest
from pathlib import Path

from wenah.core.types import (
    ProductFeatureInput,
    ProductCategory,
    FeatureType,
    DataFieldSpec,
    AlgorithmSpec,
)


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    """Get the data directory."""
    return project_root / "data"


@pytest.fixture
def sample_hiring_feature() -> ProductFeatureInput:
    """Create a sample hiring feature for testing."""
    return ProductFeatureInput(
        feature_id="test-hiring-001",
        name="Resume Screening System",
        description="AI-powered resume screening for job applicants",
        category=ProductCategory.HIRING,
        feature_type=FeatureType.ALGORITHM,
        data_fields=[
            DataFieldSpec(
                name="resume_text",
                description="Full text of resume",
                data_type="text",
                source="user_input",
                required=True,
                used_in_decisions=True,
            ),
            DataFieldSpec(
                name="years_experience",
                description="Years of relevant experience",
                data_type="numeric",
                source="derived",
                required=False,
                used_in_decisions=True,
            ),
        ],
        algorithm=AlgorithmSpec(
            name="ResumeRanker",
            type="ml_model",
            inputs=["resume_text", "job_description"],
            outputs=["match_score", "recommendation"],
            bias_testing_done=False,
        ),
        decision_impact="Determines which candidates advance to interview",
        affected_population="Job applicants",
        company_size=500,
    )


@pytest.fixture
def sample_hiring_feature_with_protected_class() -> ProductFeatureInput:
    """Create a hiring feature that collects protected class data."""
    return ProductFeatureInput(
        feature_id="test-hiring-002",
        name="Applicant Intake Form",
        description="Initial application form for job applicants",
        category=ProductCategory.HIRING,
        feature_type=FeatureType.DATA_COLLECTION,
        data_fields=[
            DataFieldSpec(
                name="name",
                description="Applicant's full name",
                data_type="text",
                source="user_input",
                required=True,
                used_in_decisions=False,
            ),
            DataFieldSpec(
                name="race",
                description="Applicant's race/ethnicity",
                data_type="categorical",
                source="user_input",
                required=False,
                used_in_decisions=True,  # This is a violation!
            ),
            DataFieldSpec(
                name="disability_status",
                description="Whether applicant has a disability",
                data_type="boolean",
                source="user_input",
                required=False,
                used_in_decisions=False,
            ),
        ],
        decision_impact="Initial screening of applicants",
        affected_population="Job applicants",
        company_size=100,
    )


@pytest.fixture
def sample_hiring_feature_with_proxy() -> ProductFeatureInput:
    """Create a hiring feature that uses proxy variables."""
    return ProductFeatureInput(
        feature_id="test-hiring-003",
        name="Candidate Scoring Algorithm",
        description="Scores candidates based on various factors",
        category=ProductCategory.HIRING,
        feature_type=FeatureType.SCORING_MODEL,
        data_fields=[
            DataFieldSpec(
                name="zip_code",
                description="Applicant's ZIP code",
                data_type="text",
                source="user_input",
                required=True,
                used_in_decisions=True,
                potential_proxy="race",
            ),
            DataFieldSpec(
                name="school_attended",
                description="Name of college/university attended",
                data_type="text",
                source="user_input",
                required=False,
                used_in_decisions=True,
            ),
        ],
        algorithm=AlgorithmSpec(
            name="CandidateScorer",
            type="ml_model",
            inputs=["zip_code", "school_attended", "gpa"],
            outputs=["candidate_score"],
            bias_testing_done=True,
        ),
        decision_impact="Ranks candidates for hiring consideration",
        affected_population="Job applicants",
    )


@pytest.fixture
def sample_video_interview_feature() -> ProductFeatureInput:
    """Create a video interview analysis feature."""
    return ProductFeatureInput(
        feature_id="test-hiring-004",
        name="Video Interview Analyzer",
        description="AI analysis of video interviews for candidate assessment",
        category=ProductCategory.HIRING,
        feature_type=FeatureType.ALGORITHM,
        data_fields=[
            DataFieldSpec(
                name="video_recording",
                description="Video recording of interview",
                data_type="binary",
                source="user_input",
                required=True,
                used_in_decisions=True,
            ),
        ],
        algorithm=AlgorithmSpec(
            name="VideoAnalyzer",
            type="ml_model",
            inputs=["video", "facial_expression", "voice_analysis", "speech_pattern"],
            outputs=["confidence_score", "communication_score"],
            bias_testing_done=False,
        ),
        decision_impact="Evaluates candidate communication and presentation",
        affected_population="Job applicants in interview stage",
    )


@pytest.fixture
def sample_compliant_feature() -> ProductFeatureInput:
    """Create a feature that should be mostly compliant."""
    return ProductFeatureInput(
        feature_id="test-hiring-005",
        name="Skills Assessment Platform",
        description="Technical skills assessment with human review",
        category=ProductCategory.HIRING,
        feature_type=FeatureType.ALGORITHM,
        data_fields=[
            DataFieldSpec(
                name="coding_submission",
                description="Code submitted by candidate",
                data_type="text",
                source="user_input",
                required=True,
                used_in_decisions=True,
            ),
            DataFieldSpec(
                name="test_scores",
                description="Scores on technical assessments",
                data_type="numeric",
                source="derived",
                required=True,
                used_in_decisions=True,
            ),
        ],
        algorithm=AlgorithmSpec(
            name="SkillsEvaluator",
            type="rule_based",
            inputs=["coding_submission", "test_answers"],
            outputs=["technical_score"],
            bias_testing_done=True,
        ),
        decision_impact="Evaluates technical skills with human oversight and review",
        affected_population="Technical job applicants",
        additional_context="All automated assessments are reviewed by human evaluators",
    )


@pytest.fixture
def sample_lending_feature() -> ProductFeatureInput:
    """Create a sample lending feature for testing."""
    return ProductFeatureInput(
        feature_id="test-lending-001",
        name="Credit Decision Engine",
        description="Automated credit approval system",
        category=ProductCategory.LENDING,
        feature_type=FeatureType.AUTOMATED_DECISION,
        data_fields=[
            DataFieldSpec(
                name="credit_score",
                description="Applicant's credit score",
                data_type="numeric",
                source="third_party",
                required=True,
                used_in_decisions=True,
            ),
            DataFieldSpec(
                name="income",
                description="Annual income",
                data_type="numeric",
                source="user_input",
                required=True,
                used_in_decisions=True,
            ),
        ],
        decision_impact="Determines credit approval and terms",
        affected_population="Credit applicants",
    )
