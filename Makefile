.PHONY: start stop restart status clean dev-backend dev-frontend check pull-models setup help

# ── Colors ──
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
CYAN   := \033[0;36m
RESET  := \033[0m

# ── Default target ──
help:
	@echo "$(CYAN)RAG Playground — Makefile targets$(RESET)"
	@echo ""
	@echo "$(GREEN)start$(RESET)          Bootstrap everything and launch"
	@echo "$(GREEN)stop$(RESET)           Stop all running services"
	@echo "$(GREEN)restart$(RESET)        Stop + start"
	@echo "$(GREEN)status$(RESET)         Show running services"
	@echo "$(GREEN)check$(RESET)          Verify prerequisites (Ollama, models, deps)"
	@echo "$(GREEN)pull-models$(RESET)    Pull required Ollama models"
	@echo "$(GREEN)setup$(RESET)          Full first-time setup"
	@echo "$(GREEN)dev-backend$(RESET)    Start only the FastAPI backend"
	@echo "$(GREEN)dev-frontend$(RESET)   Start only the React frontend"
	@echo "$(GREEN)test$(RESET)           Run all tests"
	@echo "$(GREEN)clean$(RESET)          Remove caches and build artifacts"

# ── Prerequisites ──

check:
	@echo "$(CYAN)🔍 Checking prerequisites...$(RESET)"
	@echo ""
	@# Python
	@python --version 2>/dev/null || (echo "$(RED)❌ Python 3.11+ required$(RESET)" && exit 1)
	@echo "$(GREEN)✅ Python$(RESET)"
	@# Node
	@node --version 2>/dev/null || (echo "$(YELLOW)⚠️  Node.js not found (needed for frontend)$(RESET)")
	@node --version 2>/dev/null && echo "$(GREEN)✅ Node.js$(RESET)" || true
	@# Ollama
	@curl -s http://localhost:11434/ > /dev/null 2>&1 && echo "$(GREEN)✅ Ollama running$(RESET)" || (echo "$(YELLOW)⚠️  Ollama not running on localhost:11434$(RESET)" && echo "   Start it with: ollama serve" && exit 1)
	@echo ""
	@python scripts/check_ollama.py

pull-models:
	@python scripts/pull_models.py

setup: check pull-models
	@echo ""
	@pip install -e ".[dev]" 2>/dev/null || echo "$(YELLOW)⚠️  pip install skipped (may already be installed)$(RESET)"
	@cd frontend && npm install 2>/dev/null || echo "$(YELLOW)⚠️  npm install skipped (may already be installed)$(RESET)"
	@echo ""
	@echo "$(GREEN)✅ Setup complete!$(RESET)"

# ── Development ──

dev-backend:
	@echo "$(CYAN)🚀 Starting backend on http://localhost:8000$(RESET)"
	@echo "$(CYAN)   API docs at http://localhost:8000/docs$(RESET)"
	@echo ""
	@uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend:
	@echo "$(CYAN)🎨 Starting frontend on http://localhost:5173$(RESET)"
	@echo ""
	@cd frontend && npm run dev

# ── Full stack ──

start: check
	@echo ""
	@echo "$(CYAN)🚀 Starting RAG Playground...$(RESET)"
	@echo ""
	@# Start backend in background
	@uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload > /dev/null 2>&1 &
	@echo "$(GREEN)✅ Backend started on http://localhost:8000$(RESET)"
	@# Start frontend in background
	@cd frontend && npm run dev > /dev/null 2>&1 &
	@echo "$(GREEN)✅ Frontend starting on http://localhost:5173$(RESET)"
	@echo ""
	@echo "$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@echo "$(GREEN)  RAG Playground is running!$(RESET)"
	@echo ""
	@echo "  Frontend:  http://localhost:5173"
	@echo "  Backend:   http://localhost:8000"
	@echo "  API Docs:  http://localhost:8000/docs"
	@echo "  Health:    http://localhost:8000/api/health"
	@echo ""
	@echo "  Run $(YELLOW)make stop$(RESET) to shut down"
	@echo "$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@echo ""
	@# Open browser (platform-aware)
	@(sleep 2 && (open http://localhost:5173 2>/dev/null || xdg-open http://localhost:5173 2>/dev/null || start http://localhost:5173 2>/dev/null)) &

stop:
	@echo "$(CYAN)🛑 Stopping RAG Playground...$(RESET)"
	@-pkill -f "uvicorn server.main" 2>/dev/null && echo "$(GREEN)✅ Backend stopped$(RESET)" || true
	@-pkill -f "vite" 2>/dev/null && echo "$(GREEN)✅ Frontend stopped$(RESET)" || true
	@echo "$(GREEN)✅ All services stopped$(RESET)"

restart: stop start

status:
	@echo "$(CYAN)📊 Service Status$(RESET)"
	@echo ""
	@echo -n "Ollama:    " && (curl -s http://localhost:11434/ > /dev/null 2>&1 && echo "$(GREEN)running$(RESET)" || echo "$(RED)not running$(RESET)")
	@echo -n "Backend:   " && (curl -s http://localhost:8000/api/health > /dev/null 2>&1 && echo "$(GREEN)running$(RESET)" || echo "$(RED)not running$(RESET)")
	@echo -n "Frontend:  " && (curl -s http://localhost:5173 > /dev/null 2>&1 && echo "$(GREEN)running$(RESET)" || echo "$(YELLOW)not checked$(RESET)")

# ── Tests ──

test:
	pytest

# ── Cleanup ──

clean:
	@echo "$(CYAN)🧹 Cleaning up...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .coverage coverage.xml htmlcov/ 2>/dev/null || true
	@rm -rf frontend/dist/ 2>/dev/null || true
	@echo "$(GREEN)✅ Cleaned$(RESET)"
