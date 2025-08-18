"""Add events forecasts

Revision ID: 27465c501d9a
Revises: 3063780b8c89
Create Date: 2025-07-31 21:30:48.807668

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "27465c501d9a"
down_revision: Union[str, None] = "3063780b8c89"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            ALTER TABLE events ADD COLUMN forecasts JSON NOT NULL DEFAULT '{}'
        """
    )
