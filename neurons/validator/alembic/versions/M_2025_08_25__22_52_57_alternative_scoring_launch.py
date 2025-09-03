"""Alternative scoring launch

Revision ID: 55ade8b18263
Revises: 00ad2c49ac31
Create Date: 2025-08-25 22:52:57.837125

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "55ade8b18263"
down_revision: Union[str, None] = "00ad2c49ac31"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            UPDATE
                scores
            SET
                alternative_processed = TRUE
            WHERE
                alternative_processed = FALSE
                AND processed = TRUE
        """
    )
