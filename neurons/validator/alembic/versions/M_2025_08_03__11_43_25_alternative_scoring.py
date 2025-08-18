"""Alternative scoring

Revision ID: c6848f2e0e3f
Revises: 27465c501d9a
Create Date: 2025-08-03 11:43:25.274065

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c6848f2e0e3f"
down_revision: Union[str, None] = "27465c501d9a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            ALTER TABLE events ADD COLUMN alternative_scored BOOLEAN NOT NULL DEFAULT false
        """
    )

    op.execute(
        """
            ALTER TABLE scores ADD COLUMN alternative_metagraph_score REAL
        """
    )

    op.execute(
        """
            ALTER TABLE scores ADD COLUMN alternative_other_data JSON
        """
    )
