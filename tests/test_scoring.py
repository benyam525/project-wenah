"""
Tests for the unified scoring system and compliance engine.

Tests cover:
- ScoringEngine: score calculation, weight adjustment, confidence intervals
- ScoreExplainer: explanation generation at different detail levels
- ComplianceEngine: feature and product assessment orchestration
"""

from unittest.mock import Mock, patch

import pytest

from wenah.core.engine import (
    AssessmentConfig,
    ComplianceEngine,
    FeatureAnalysis,
)
from wenah.core.scoring import (
    ScoreExplainer,
    ScoreSource,
    ScoringEngine,
    get_score_explainer,
    get_scoring_engine,
)
from wenah.core.types import (
    CategoryBreakdown,
    ComponentScore,
    RAGResponse,
    RiskLevel,
    RuleEvaluation,
    RuleResult,
    UnifiedRiskScore,
)


class TestScoringEngine:
    """Tests for ScoringEngine class."""

    @pytest.fixture
    def scoring_engine(self):
        """Create scoring engine with default settings."""
        return ScoringEngine(
            rule_base_weight=0.6,
            llm_base_weight=0.4,
            high_confidence_threshold=0.8,
            low_confidence_threshold=0.5,
        )

    @pytest.fixture
    def sample_violation_evaluation(self):
        """Create a sample violation rule evaluation."""
        return RuleEvaluation(
            rule_id="rule-001",
            rule_name="Protected Class Usage",
            result=RuleResult.VIOLATION,
            risk_score=85.0,
            confidence=1.0,
            law_references=["Title VII"],
            recommendations=["Remove protected class data from decision-making"],
            affected_fields=["race"],
            escalate_to_llm=False,
        )

    @pytest.fixture
    def sample_potential_violation_evaluation(self):
        """Create a sample potential violation rule evaluation."""
        return RuleEvaluation(
            rule_id="rule-002",
            rule_name="Proxy Variable Detection",
            result=RuleResult.POTENTIAL_VIOLATION,
            risk_score=55.0,
            confidence=0.7,
            law_references=["Title VII"],
            recommendations=["Review zip code usage for disparate impact"],
            affected_fields=["zip_code"],
            escalate_to_llm=True,
            llm_context={"question": "Does zip code correlate with race?"},
        )

    @pytest.fixture
    def sample_compliant_evaluation(self):
        """Create a sample compliant rule evaluation."""
        return RuleEvaluation(
            rule_id="rule-003",
            rule_name="Human Oversight Check",
            result=RuleResult.COMPLIANT,
            risk_score=10.0,
            confidence=1.0,
            law_references=["ADA"],
            recommendations=[],
            affected_fields=[],
            escalate_to_llm=False,
        )

    @pytest.fixture
    def sample_rag_response(self):
        """Create a sample RAG response."""
        return RAGResponse(
            analysis="The feature shows potential disparate impact concerns.",
            confidence_score=0.75,
            cited_sources=["Title VII", "Griggs v. Duke Power"],
            risk_factors=["Proxy variable usage", "No bias testing"],
            mitigating_factors=["Human review process"],
            recommendation="Conduct disparate impact analysis before deployment",
            requires_human_review=True,
        )

    # Test _calculate_rule_score

    def test_rule_score_no_evaluations(self, scoring_engine):
        """Test rule score calculation with no evaluations."""
        score = scoring_engine._calculate_rule_score([])

        assert score.source == ScoreSource.RULE_ENGINE.value
        assert score.raw_score == 0.0
        assert score.confidence == 1.0
        assert "No applicable rules" in score.explanation

    def test_rule_score_critical_violation(
        self,
        scoring_engine,
        sample_violation_evaluation,
    ):
        """Test rule score with critical violation (score >= 80)."""
        score = scoring_engine._calculate_rule_score([sample_violation_evaluation])

        assert score.raw_score == 85.0
        assert score.confidence == 1.0
        assert "Critical violation" in score.explanation

    def test_rule_score_multiple_violations(self, scoring_engine):
        """Test rule score with multiple non-critical violations."""
        evals = [
            RuleEvaluation(
                rule_id="r1",
                rule_name="Rule 1",
                result=RuleResult.VIOLATION,
                risk_score=60.0,
                confidence=0.9,
                law_references=["Title VII"],
                recommendations=["Fix issue 1"],
                affected_fields=["field1"],
                escalate_to_llm=False,
            ),
            RuleEvaluation(
                rule_id="r2",
                rule_name="Rule 2",
                result=RuleResult.VIOLATION,
                risk_score=50.0,
                confidence=0.8,
                law_references=["ADA"],
                recommendations=["Fix issue 2"],
                affected_fields=["field2"],
                escalate_to_llm=False,
            ),
        ]

        score = scoring_engine._calculate_rule_score(evals)

        # Weighted average: (60*0.9 + 50*0.8) / (0.9+0.8) = 94/1.7 = 55.29
        assert 50 < score.raw_score < 60
        assert "2 violation(s)" in score.explanation

    def test_rule_score_potential_violations_only(
        self,
        scoring_engine,
        sample_potential_violation_evaluation,
    ):
        """Test rule score with only potential violations."""
        score = scoring_engine._calculate_rule_score([sample_potential_violation_evaluation])

        assert score.raw_score == 55.0
        assert score.confidence < 1.0  # Should be reduced for potential violations
        assert "potential violation" in score.explanation.lower()

    def test_rule_score_compliant_only(
        self,
        scoring_engine,
        sample_compliant_evaluation,
    ):
        """Test rule score with only compliant evaluations."""
        score = scoring_engine._calculate_rule_score([sample_compliant_evaluation])

        assert score.raw_score < 20  # Low risk for compliant
        assert score.confidence == 1.0
        assert "No violations" in score.explanation

    # Test _calculate_llm_score

    def test_llm_score_basic(self, scoring_engine, sample_rag_response):
        """Test LLM score calculation."""
        score = scoring_engine._calculate_llm_score(sample_rag_response)

        assert score.source == ScoreSource.LLM_ANALYSIS.value
        assert score.confidence == 0.75
        assert 0 <= score.raw_score <= 100
        assert "risk factor" in score.explanation.lower()

    def test_llm_score_high_risk(self, scoring_engine):
        """Test LLM score with high risk factors."""
        response = RAGResponse(
            analysis="Multiple serious violations detected.",
            confidence_score=0.9,
            cited_sources=["Title VII"],
            risk_factors=["Factor 1", "Factor 2", "Factor 3", "Factor 4"],
            mitigating_factors=[],
            recommendation="Major remediation required",
            requires_human_review=True,
        )

        score = scoring_engine._calculate_llm_score(response)

        # Base 30 + 4*10 + 15 (human review) = 85
        assert score.raw_score >= 70

    def test_llm_score_low_risk(self, scoring_engine):
        """Test LLM score with mitigating factors."""
        response = RAGResponse(
            analysis="Generally compliant with minor concerns.",
            confidence_score=0.85,
            cited_sources=["Title VII"],
            risk_factors=[],
            mitigating_factors=["Bias testing done", "Human oversight", "Appeals process"],
            recommendation="Minor adjustments recommended",
            requires_human_review=False,
        )

        score = scoring_engine._calculate_llm_score(response)

        # Base 30 - 3*8 = 6, but clamped to 0 minimum
        assert score.raw_score <= 30

    # Test _has_critical_violation

    def test_has_critical_violation_true(
        self,
        scoring_engine,
        sample_violation_evaluation,
    ):
        """Test critical violation detection when present."""
        assert scoring_engine._has_critical_violation([sample_violation_evaluation])

    def test_has_critical_violation_false(
        self,
        scoring_engine,
        sample_potential_violation_evaluation,
    ):
        """Test critical violation detection when absent."""
        assert not scoring_engine._has_critical_violation([sample_potential_violation_evaluation])

    # Test _adjust_weights

    def test_adjust_weights_critical_violation(self, scoring_engine):
        """Test weight adjustment for critical violations."""
        rule_score = ComponentScore(
            source=ScoreSource.RULE_ENGINE.value,
            raw_score=85.0,
            confidence=1.0,
            weight=0.6,
            weighted_score=51.0,
            explanation="Critical violation",
        )
        llm_score = ComponentScore(
            source=ScoreSource.LLM_ANALYSIS.value,
            raw_score=70.0,
            confidence=0.8,
            weight=0.4,
            weighted_score=22.4,
            explanation="LLM analysis",
        )

        adjusted = scoring_engine._adjust_weights(
            [rule_score, llm_score],
            has_critical=True,
            rule_evaluations=[],
            rag_response=None,
        )

        # Rule weight should increase, LLM should decrease
        rule_adjusted = next(s for s in adjusted if s.source == ScoreSource.RULE_ENGINE.value)
        llm_adjusted = next(s for s in adjusted if s.source == ScoreSource.LLM_ANALYSIS.value)

        assert rule_adjusted.weight > 0.6
        assert llm_adjusted.weight < 0.4

    def test_adjust_weights_low_confidence_llm(self, scoring_engine):
        """Test weight adjustment for low confidence LLM."""
        llm_score = ComponentScore(
            source=ScoreSource.LLM_ANALYSIS.value,
            raw_score=50.0,
            confidence=0.3,  # Low confidence
            weight=0.4,
            weighted_score=6.0,
            explanation="Low confidence analysis",
        )

        adjusted = scoring_engine._adjust_weights(
            [llm_score],
            has_critical=False,
            rule_evaluations=[],
            rag_response=None,
        )

        assert adjusted[0].weight < 0.4  # Should be reduced

    # Test _combine_scores

    def test_combine_scores_empty(self, scoring_engine):
        """Test combining empty score list."""
        assert scoring_engine._combine_scores([]) == 0.0

    def test_combine_scores_single(self, scoring_engine):
        """Test combining single score."""
        score = ComponentScore(
            source=ScoreSource.RULE_ENGINE.value,
            raw_score=50.0,
            confidence=1.0,
            weight=1.0,
            weighted_score=50.0,
            explanation="Test",
        )

        result = scoring_engine._combine_scores([score])
        assert result == 50.0

    def test_combine_scores_weighted_average(self, scoring_engine):
        """Test weighted average combination."""
        scores = [
            ComponentScore(
                source=ScoreSource.RULE_ENGINE.value,
                raw_score=60.0,
                confidence=1.0,
                weight=0.6,
                weighted_score=36.0,
                explanation="Rule score",
            ),
            ComponentScore(
                source=ScoreSource.LLM_ANALYSIS.value,
                raw_score=40.0,
                confidence=1.0,
                weight=0.4,
                weighted_score=16.0,
                explanation="LLM score",
            ),
        ]

        result = scoring_engine._combine_scores(scores)

        # (36 + 16) / 1.0 = 52
        assert 50 < result < 55

    # Test _calculate_confidence_interval

    def test_confidence_interval_high_confidence(self, scoring_engine):
        """Test confidence interval with high confidence scores."""
        scores = [
            ComponentScore(
                source=ScoreSource.RULE_ENGINE.value,
                raw_score=50.0,
                confidence=0.95,
                weight=0.6,
                weighted_score=28.5,
                explanation="High confidence",
            ),
        ]

        low, high = scoring_engine._calculate_confidence_interval(scores, 50.0)

        # Should have narrow margin
        assert high - low < 5

    def test_confidence_interval_low_confidence(self, scoring_engine):
        """Test confidence interval with low confidence scores."""
        scores = [
            ComponentScore(
                source=ScoreSource.RULE_ENGINE.value,
                raw_score=50.0,
                confidence=0.5,
                weight=0.6,
                weighted_score=15.0,
                explanation="Low confidence",
            ),
        ]

        low, high = scoring_engine._calculate_confidence_interval(scores, 50.0)

        # Should have wider margin
        assert high - low >= 10

    def test_confidence_interval_mixed_signals(self, scoring_engine):
        """Test confidence interval with mixed signals."""
        scores = [
            ComponentScore(
                source=ScoreSource.RULE_ENGINE.value,
                raw_score=80.0,
                confidence=0.9,
                weight=0.6,
                weighted_score=43.2,
                explanation="High risk",
            ),
            ComponentScore(
                source=ScoreSource.LLM_ANALYSIS.value,
                raw_score=30.0,
                confidence=0.8,
                weight=0.4,
                weighted_score=9.6,
                explanation="Low risk",
            ),
        ]

        low, high = scoring_engine._calculate_confidence_interval(scores, 55.0)

        # Spread > 30, should add extra margin
        assert high - low >= 5

    # Test _score_to_risk_level

    def test_score_to_risk_level_critical(self, scoring_engine):
        """Test critical risk level mapping."""
        assert scoring_engine._score_to_risk_level(85) == RiskLevel.CRITICAL
        assert scoring_engine._score_to_risk_level(100) == RiskLevel.CRITICAL

    def test_score_to_risk_level_high(self, scoring_engine):
        """Test high risk level mapping."""
        assert scoring_engine._score_to_risk_level(60) == RiskLevel.HIGH
        assert scoring_engine._score_to_risk_level(79) == RiskLevel.HIGH

    def test_score_to_risk_level_medium(self, scoring_engine):
        """Test medium risk level mapping."""
        assert scoring_engine._score_to_risk_level(40) == RiskLevel.MEDIUM
        assert scoring_engine._score_to_risk_level(59) == RiskLevel.MEDIUM

    def test_score_to_risk_level_low(self, scoring_engine):
        """Test low risk level mapping."""
        assert scoring_engine._score_to_risk_level(20) == RiskLevel.LOW
        assert scoring_engine._score_to_risk_level(39) == RiskLevel.LOW

    def test_score_to_risk_level_minimal(self, scoring_engine):
        """Test minimal risk level mapping."""
        assert scoring_engine._score_to_risk_level(0) == RiskLevel.MINIMAL
        assert scoring_engine._score_to_risk_level(19) == RiskLevel.MINIMAL

    # Test _check_human_review_needed

    def test_human_review_high_risk(self, scoring_engine):
        """Test human review for high risk score."""
        scores = [
            ComponentScore(
                source=ScoreSource.RULE_ENGINE.value,
                raw_score=70.0,
                confidence=0.9,
                weight=0.6,
                weighted_score=37.8,
                explanation="High risk",
            ),
        ]

        needed, reasons = scoring_engine._check_human_review_needed(scores, [], None)

        assert needed
        assert any("risk score" in r.lower() for r in reasons)

    def test_human_review_low_confidence(self, scoring_engine):
        """Test human review for low confidence."""
        scores = [
            ComponentScore(
                source=ScoreSource.RULE_ENGINE.value,
                raw_score=30.0,
                confidence=0.3,
                weight=0.6,
                weighted_score=5.4,
                explanation="Low confidence",
            ),
        ]

        needed, reasons = scoring_engine._check_human_review_needed(scores, [], None)

        assert needed
        assert any("confidence" in r.lower() for r in reasons)

    def test_human_review_escalated_rules(
        self,
        scoring_engine,
        sample_potential_violation_evaluation,
    ):
        """Test human review for escalated rules (requires medium+ risk score)."""
        # Score must produce overall >= 40 after weighting
        # With confidence=0.9, weight=1.0: overall = 50 * 0.9 * 1.0 / 1.0 = 45
        scores = [
            ComponentScore(
                source=ScoreSource.RULE_ENGINE.value,
                raw_score=50.0,
                confidence=0.9,
                weight=1.0,
                weighted_score=45.0,
                explanation="Medium risk",
            ),
        ]

        needed, reasons = scoring_engine._check_human_review_needed(
            scores, [sample_potential_violation_evaluation], None
        )

        assert needed
        assert any("nuanced analysis" in r.lower() for r in reasons)

    def test_human_review_conflicting_signals(self, scoring_engine):
        """Test human review for conflicting rule and LLM scores."""
        scores = [
            ComponentScore(
                source=ScoreSource.RULE_ENGINE.value,
                raw_score=80.0,
                confidence=0.9,
                weight=0.6,
                weighted_score=43.2,
                explanation="High risk",
            ),
            ComponentScore(
                source=ScoreSource.LLM_ANALYSIS.value,
                raw_score=30.0,
                confidence=0.8,
                weight=0.4,
                weighted_score=9.6,
                explanation="Low risk",
            ),
        ]

        needed, reasons = scoring_engine._check_human_review_needed(scores, [], None)

        assert needed
        assert any("conflicting" in r.lower() for r in reasons)

    # Test _calculate_category_breakdown

    def test_category_breakdown_employment(self, scoring_engine):
        """Test category breakdown for employment rules."""
        evals = [
            RuleEvaluation(
                rule_id="r1",
                rule_name="Title VII Rule",
                result=RuleResult.VIOLATION,
                risk_score=70.0,
                confidence=0.9,
                law_references=["Title VII"],
                recommendations=["Fix"],
                affected_fields=["race"],
                escalate_to_llm=False,
            ),
        ]

        breakdown = scoring_engine._calculate_category_breakdown(evals)

        assert breakdown.employment > 0
        assert breakdown.housing == 0
        assert breakdown.consumer == 0

    def test_category_breakdown_mixed(self, scoring_engine):
        """Test category breakdown with mixed categories."""
        evals = [
            RuleEvaluation(
                rule_id="r1",
                rule_name="Employment Rule",
                result=RuleResult.VIOLATION,
                risk_score=60.0,
                confidence=1.0,
                law_references=["Title VII"],
                recommendations=["Fix"],
                affected_fields=[],
                escalate_to_llm=False,
            ),
            RuleEvaluation(
                rule_id="r2",
                rule_name="Consumer Rule",
                result=RuleResult.VIOLATION,
                risk_score=50.0,
                confidence=0.8,
                law_references=["ECOA"],
                recommendations=["Fix"],
                affected_fields=[],
                escalate_to_llm=False,
            ),
        ]

        breakdown = scoring_engine._calculate_category_breakdown(evals)

        assert breakdown.employment > 0
        assert breakdown.consumer > 0
        assert breakdown.overall == max(breakdown.employment, breakdown.consumer)

    # Test calculate_unified_score (integration)

    def test_unified_score_rules_only(
        self,
        scoring_engine,
        sample_violation_evaluation,
    ):
        """Test unified score with rules only."""
        score = scoring_engine.calculate_unified_score(
            rule_evaluations=[sample_violation_evaluation],
            rag_response=None,
        )

        assert isinstance(score, UnifiedRiskScore)
        assert score.overall_score >= 80  # Critical violation
        assert score.risk_level == RiskLevel.CRITICAL
        assert len(score.component_scores) == 1
        assert score.requires_human_review  # High risk triggers review

    def test_unified_score_with_rag(
        self,
        scoring_engine,
        sample_potential_violation_evaluation,
        sample_rag_response,
    ):
        """Test unified score with RAG analysis."""
        score = scoring_engine.calculate_unified_score(
            rule_evaluations=[sample_potential_violation_evaluation],
            rag_response=sample_rag_response,
        )

        assert isinstance(score, UnifiedRiskScore)
        assert len(score.component_scores) == 2  # Rule + LLM
        assert score.primary_concerns  # Should have concerns
        assert score.recommendations  # Should have recommendations

    def test_unified_score_compliant(
        self,
        scoring_engine,
        sample_compliant_evaluation,
    ):
        """Test unified score for compliant feature."""
        score = scoring_engine.calculate_unified_score(
            rule_evaluations=[sample_compliant_evaluation],
        )

        assert score.overall_score < 30
        assert score.risk_level in [RiskLevel.MINIMAL, RiskLevel.LOW]


class TestScoreExplainer:
    """Tests for ScoreExplainer class."""

    @pytest.fixture
    def explainer(self):
        """Create score explainer."""
        return ScoreExplainer()

    @pytest.fixture
    def sample_unified_score(self):
        """Create sample unified score for explanation."""
        return UnifiedRiskScore(
            overall_score=65.0,
            risk_level=RiskLevel.HIGH,
            component_scores=[
                ComponentScore(
                    source=ScoreSource.RULE_ENGINE.value,
                    raw_score=70.0,
                    confidence=0.9,
                    weight=0.6,
                    weighted_score=37.8,
                    explanation="2 violations detected",
                ),
                ComponentScore(
                    source=ScoreSource.LLM_ANALYSIS.value,
                    raw_score=55.0,
                    confidence=0.8,
                    weight=0.4,
                    weighted_score=17.6,
                    explanation="LLM identified 2 risk factors",
                ),
            ],
            category_breakdown=CategoryBreakdown(
                employment=65.0,
                housing=0.0,
                consumer=0.0,
                overall=65.0,
            ),
            confidence_interval=(60.0, 70.0),
            primary_concerns=[
                "Protected class data usage",
                "Lack of bias testing",
            ],
            recommendations=[
                "Remove race field from decision inputs",
                "Conduct bias testing",
                "Add human oversight",
            ],
            requires_human_review=True,
        )

    def test_explain_brief(self, explainer, sample_unified_score):
        """Test brief explanation."""
        explanation = explainer.explain_score(sample_unified_score, detail_level="brief")

        assert "65/100" in explanation
        assert "HIGH" in explanation
        assert "human review recommended" in explanation.lower()
        # Brief should have fewer sections
        assert "Category" not in explanation

    def test_explain_standard(self, explainer, sample_unified_score):
        """Test standard explanation."""
        explanation = explainer.explain_score(sample_unified_score, detail_level="standard")

        assert "65/100" in explanation
        assert "Score Components" in explanation
        assert "Primary Concerns" in explanation
        assert "Recommended Actions" in explanation

    def test_explain_detailed(self, explainer, sample_unified_score):
        """Test detailed explanation."""
        explanation = explainer.explain_score(sample_unified_score, detail_level="detailed")

        assert "65/100" in explanation
        assert "Score Components" in explanation
        assert "Primary Concerns" in explanation
        assert "Risk by Category" in explanation
        assert "Confidence Interval" in explanation
        assert "60" in explanation and "70" in explanation  # Interval values

    def test_explain_no_concerns(self, explainer):
        """Test explanation with no concerns."""
        score = UnifiedRiskScore(
            overall_score=15.0,
            risk_level=RiskLevel.MINIMAL,
            component_scores=[
                ComponentScore(
                    source=ScoreSource.RULE_ENGINE.value,
                    raw_score=15.0,
                    confidence=1.0,
                    weight=1.0,
                    weighted_score=15.0,
                    explanation="No violations detected",
                ),
            ],
            category_breakdown=CategoryBreakdown(
                employment=0.0,
                housing=0.0,
                consumer=0.0,
                overall=0.0,
            ),
            confidence_interval=(10.0, 20.0),
            primary_concerns=[],
            recommendations=[],
            requires_human_review=False,
        )

        explanation = explainer.explain_score(score, detail_level="standard")

        assert "15/100" in explanation
        assert "MINIMAL" in explanation
        assert "human review recommended" not in explanation.lower()


class TestComplianceEngine:
    """Tests for ComplianceEngine class."""

    @pytest.fixture
    def mock_rule_engine(self):
        """Create mock rule engine."""
        engine = Mock()
        engine.evaluate.return_value = [
            RuleEvaluation(
                rule_id="r1",
                rule_name="Test Rule",
                result=RuleResult.POTENTIAL_VIOLATION,
                risk_score=45.0,
                confidence=0.8,
                law_references=["Title VII"],
                recommendations=["Review for compliance"],
                affected_fields=["field1"],
                escalate_to_llm=False,
            ),
        ]
        return engine

    @pytest.fixture
    def mock_rag_pipeline(self):
        """Create mock RAG pipeline."""
        pipeline = Mock()
        mock_result = Mock()
        mock_result.response = RAGResponse(
            analysis="Analysis complete.",
            confidence_score=0.7,
            cited_sources=["Title VII"],
            risk_factors=["Factor 1"],
            mitigating_factors=[],
            recommendation="Review recommended",
            requires_human_review=False,
        )
        mock_result.retrieval = Mock()
        mock_result.raw_llm_response = "{}"
        mock_result.tokens_used = {"input_tokens": 100, "output_tokens": 50}
        pipeline.analyze.return_value = mock_result
        return pipeline

    @pytest.fixture
    def mock_employment_processor(self):
        """Create mock employment processor."""
        with patch("wenah.core.engine.get_employment_processor") as mock:
            processor = Mock()
            processor.analyze_feature.return_value = {
                "status": "analyzed",
                "findings": [],
                "recommendations": [],
            }
            mock.return_value = processor
            yield processor

    @pytest.fixture
    def compliance_engine(
        self,
        mock_rule_engine,
        mock_rag_pipeline,
        mock_employment_processor,
    ):
        """Create compliance engine with mocks."""
        return ComplianceEngine(
            rule_engine=mock_rule_engine,
            rag_pipeline=mock_rag_pipeline,
            scoring_engine=ScoringEngine(),
            score_explainer=ScoreExplainer(),
        )

    def test_assess_feature_basic(
        self,
        compliance_engine,
        sample_hiring_feature,
    ):
        """Test basic feature assessment."""
        config = AssessmentConfig(
            include_llm_analysis=False,
            include_category_analysis=False,
        )

        analysis = compliance_engine.assess_feature(sample_hiring_feature, config)

        assert isinstance(analysis, FeatureAnalysis)
        assert analysis.feature == sample_hiring_feature
        assert len(analysis.rule_evaluations) > 0
        assert analysis.unified_score is not None
        assert analysis.explanation

    def test_assess_feature_with_category_analysis(
        self,
        compliance_engine,
        sample_hiring_feature,
        mock_employment_processor,
    ):
        """Test feature assessment with category analysis."""
        config = AssessmentConfig(
            include_llm_analysis=False,
            include_category_analysis=True,
        )

        analysis = compliance_engine.assess_feature(sample_hiring_feature, config)

        assert analysis.category_analysis is not None
        mock_employment_processor.analyze_feature.assert_called_once()

    def test_assess_feature_with_llm(
        self,
        compliance_engine,
        sample_hiring_feature,
        mock_rag_pipeline,
    ):
        """Test feature assessment with LLM analysis."""
        # Modify rule engine to return escalated rule
        compliance_engine.rule_engine.evaluate.return_value = [
            RuleEvaluation(
                rule_id="r1",
                rule_name="Test Rule",
                result=RuleResult.POTENTIAL_VIOLATION,
                risk_score=65.0,  # High enough to trigger LLM
                confidence=0.8,
                law_references=["Title VII"],
                recommendations=["Review"],
                affected_fields=[],
                escalate_to_llm=True,
            ),
        ]

        config = AssessmentConfig(
            include_llm_analysis=True,
            include_category_analysis=False,
        )

        analysis = compliance_engine.assess_feature(sample_hiring_feature, config)

        assert analysis.rag_result is not None

    def test_assess_product(
        self,
        compliance_engine,
        sample_hiring_feature,
        sample_compliant_feature,
    ):
        """Test product assessment with multiple features."""
        config = AssessmentConfig(
            include_llm_analysis=False,
            include_category_analysis=False,
        )

        response = compliance_engine.assess_product(
            product_name="Test Product",
            features=[sample_hiring_feature, sample_compliant_feature],
            config=config,
        )

        assert response.product_name == "Test Product"
        assert len(response.feature_assessments) == 2
        assert response.overall_risk_score >= 0
        assert response.executive_summary
        assert response.assessment_id

    def test_quick_assess(
        self,
        compliance_engine,
        sample_hiring_feature,
    ):
        """Test quick assessment (no LLM)."""
        result = compliance_engine.quick_assess(sample_hiring_feature)

        assert "feature_id" in result
        assert "risk_score" in result
        assert "risk_level" in result
        assert "violations_count" in result
        assert result["risk_score"] >= 0

    def test_assess_feature_rag_failure_handled(
        self,
        compliance_engine,
        sample_hiring_feature,
    ):
        """Test that RAG failures don't break assessment."""
        # Make RAG pipeline raise an exception
        compliance_engine.rag_pipeline.analyze.side_effect = Exception("API Error")

        # Make rule engine return escalated rule to trigger RAG
        compliance_engine.rule_engine.evaluate.return_value = [
            RuleEvaluation(
                rule_id="r1",
                rule_name="Test Rule",
                result=RuleResult.POTENTIAL_VIOLATION,
                risk_score=65.0,
                confidence=0.8,
                law_references=["Title VII"],
                recommendations=["Review"],
                affected_fields=[],
                escalate_to_llm=True,
            ),
        ]

        config = AssessmentConfig(include_llm_analysis=True)

        # Should not raise, should handle gracefully
        analysis = compliance_engine.assess_feature(sample_hiring_feature, config)

        assert analysis.unified_score is not None
        assert analysis.rag_result is None  # RAG failed but assessment continued


class TestSingletonGetters:
    """Tests for singleton getter functions."""

    def test_get_scoring_engine(self):
        """Test singleton scoring engine getter."""
        engine1 = get_scoring_engine()
        engine2 = get_scoring_engine()

        assert engine1 is engine2
        assert isinstance(engine1, ScoringEngine)

    def test_get_score_explainer(self):
        """Test singleton score explainer getter."""
        explainer1 = get_score_explainer()
        explainer2 = get_score_explainer()

        assert explainer1 is explainer2
        assert isinstance(explainer1, ScoreExplainer)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def scoring_engine(self):
        """Create scoring engine for edge case tests."""
        return ScoringEngine()

    def test_zero_confidence_evaluation(self, scoring_engine):
        """Test handling of zero confidence evaluation."""
        eval = RuleEvaluation(
            rule_id="r1",
            rule_name="Zero Conf",
            result=RuleResult.VIOLATION,
            risk_score=50.0,
            confidence=0.0,  # Edge case
            law_references=["Title VII"],
            recommendations=["Fix"],
            affected_fields=[],
            escalate_to_llm=False,
        )

        score = scoring_engine._calculate_rule_score([eval])

        # Should handle gracefully
        assert score.raw_score >= 0

    def test_score_clamping(self, scoring_engine):
        """Test that scores are properly clamped to 0-100."""
        # Very high risk factors
        response = RAGResponse(
            analysis="Test",
            confidence_score=1.0,
            cited_sources=[],
            risk_factors=["F" + str(i) for i in range(20)],  # 20 factors
            mitigating_factors=[],
            recommendation="Test",
            requires_human_review=True,
        )

        score = scoring_engine._calculate_llm_score(response)

        assert 0 <= score.raw_score <= 100

    def test_empty_law_references(self, scoring_engine):
        """Test handling of empty law references."""
        eval = RuleEvaluation(
            rule_id="r1",
            rule_name="No Laws",
            result=RuleResult.VIOLATION,
            risk_score=50.0,
            confidence=0.8,
            law_references=[],  # Empty
            recommendations=["Fix"],
            affected_fields=[],
            escalate_to_llm=False,
        )

        breakdown = scoring_engine._calculate_category_breakdown([eval])

        # Should not crash, all categories should be 0
        assert breakdown.employment == 0
        assert breakdown.housing == 0
        assert breakdown.consumer == 0
