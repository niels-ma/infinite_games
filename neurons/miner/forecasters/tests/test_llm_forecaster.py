from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from forecasting_tools import BinaryQuestion

from neurons.miner.forecasters.llm_forecaster import LLMForecaster, LLMForecasterWithSN13
from neurons.miner.models.event import MinerEvent
from neurons.miner.models.sn13 import SN13Data, SN13Response


@pytest.fixture
def llm_forecaster():
    mock_event = Mock(spec=MinerEvent)
    mock_event.get_description = Mock(return_value="test_event")
    return LLMForecaster(event=mock_event, logger=Mock())


@pytest.fixture
def llm_forecaster_with_sn13():
    mock_event = Mock(spec=MinerEvent)
    mock_event.get_description = Mock(return_value="test_event")
    return LLMForecasterWithSN13(event=mock_event, logger=Mock(), sn13_client=Mock())


@pytest.mark.asyncio
async def test_llm_forecaster_successful_prediction(llm_forecaster):
    mock_report = Mock()
    mock_report.prediction = 0.75
    llm_forecaster.bot.forecast_questions = AsyncMock(return_value=[mock_report])

    probability, reasoning = await llm_forecaster._run()

    assert llm_forecaster.bot.forecast_questions.called

    assert isinstance(probability, float)
    assert probability == 0.75

    assert reasoning is None


@pytest.mark.asyncio
async def test_llm_forecaster_handles_exception(llm_forecaster):
    llm_forecaster.bot.forecast_questions = AsyncMock(side_effect=Exception("Error"))

    probability, reasoning = await llm_forecaster._run()

    assert isinstance(probability, float)
    assert probability == 0.5

    assert reasoning is None


@pytest.mark.asyncio
async def test_llm_forecaster_get_question(llm_forecaster):
    question = await llm_forecaster._get_question()
    assert isinstance(question, BinaryQuestion)
    assert question.question_text == "test_event"


@pytest.mark.asyncio
async def test_llm_forecaster_with_sn13_get_question(llm_forecaster_with_sn13):
    llm_forecaster_with_sn13.sn13_client.get_on_demand_data_with_gpt = AsyncMock(
        return_value=SN13Response(
            data=[
                SN13Data(
                    uri="test_uri",
                    datetime=datetime.now(timezone.utc),
                    source="X",
                    content="test_content_1",
                    label=None,
                ),
                SN13Data(
                    uri="test_uri",
                    datetime=datetime.now(timezone.utc),
                    source="X",
                    content="test_content_2",
                    label=None,
                ),
            ],
            status="status",
            meta={"test": "test"},
        )
    )

    question = await llm_forecaster_with_sn13._get_question()
    assert isinstance(question, BinaryQuestion)
    assert question.question_text == "test_event"
    assert question.background_info == "test_content_1. test_content_2"


@pytest.mark.asyncio
async def test_llm_forecaster_with_sn13_exception(llm_forecaster_with_sn13):
    llm_forecaster_with_sn13.sn13_client.get_on_demand_data_with_gpt = AsyncMock(
        side_effect=Exception("Error")
    )

    result = await llm_forecaster_with_sn13._run()
    assert result == (0.5, None)
