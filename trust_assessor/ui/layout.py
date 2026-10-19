"""Application shell: header, navigation drawer and responsive content container."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from nicegui import ui

from trust_assessor import __version__
from trust_assessor.ui import theme

NAV = [
    ("dashboard", "Dashboard", "dashboard", "/"),
    ("search", "Product Search", "search", "/search"),
    ("compare", "Compare", "compare_arrows", "/compare"),
    ("history", "Assessment History", "history", "/history"),
    ("models", "Model Insights", "psychology", "/models"),
    ("methodology", "Methodology", "menu_book", "/methodology"),
]


@contextmanager
def frame(title: str, active: str) -> Iterator[ui.column]:
    ui.colors(primary=theme.PRIMARY, positive="#16a34a", warning="#d97706", negative="#dc2626", secondary="#64748b")
    ui.add_css(theme.CSS)
    ui.page_title(f"{title} · Product Trust Assessor")

    with ui.left_drawer(value=True, bordered=True).props("width=250 breakpoint=1024").classes("bg-white") as drawer:
        with ui.column().classes("gap-1 w-full tc-nav"):
            with ui.row().classes("items-center gap-2 px-2 py-3 no-wrap"):
                ui.icon("verified_user", size="30px").classes("text-indigo-600")
                with ui.column().classes("gap-0"):
                    ui.label("Trust Assessor").classes("text-base font-bold text-slate-900")
                    ui.label("E-commerce product reliability").classes("text-xs text-slate-500")
            for key, label, icon, href in NAV:
                is_active = key == active
                with ui.link(target=href).classes("no-underline w-full"):
                    with ui.row().classes(
                        "items-center gap-3 px-3 py-2 no-wrap w-full "
                        + ("bg-indigo-50 text-indigo-700" if is_active else "text-slate-700 hover:bg-slate-100")
                    ).style("border-radius:10px"):
                        ui.icon(icon, size="20px")
                        ui.label(label).classes("text-sm " + ("font-semibold" if is_active else ""))
            ui.separator().classes("my-3")
            

    with ui.header(elevated=False).classes("bg-white text-slate-900 items-center gap-2 border-b").style("border-color:#e2e8f0"):
        ui.button(icon="menu", on_click=drawer.toggle).props("flat round dense").classes("lg:hidden text-slate-700")
        ui.label(title).classes("text-lg font-semibold")
        ui.space()
        ui.label("Heuristic screening tool: scores are not verdicts").classes("text-xs text-slate-500 hidden md:block")

    with ui.column().classes("w-full max-w-screen-xl mx-auto p-3 md:p-6 gap-5") as content:
        yield content
