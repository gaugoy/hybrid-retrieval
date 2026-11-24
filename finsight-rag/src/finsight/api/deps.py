"""Dependency helpers: the app-wide Container, and per-request containers for overrides."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from finsight.container import Container


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]
