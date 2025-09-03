"""Fix forecasts JSONs

Revision ID: e71c07407574
Revises: 55ade8b18263
Create Date: 2025-08-26 17:14:38.492115

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e71c07407574"
down_revision: Union[str, None] = "55ade8b18263"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            UPDATE
                events
            SET
                forecasts = REPLACE(forecasts, "'", '"')
            WHERE
                json_valid(forecasts) = 0
        """
    )
