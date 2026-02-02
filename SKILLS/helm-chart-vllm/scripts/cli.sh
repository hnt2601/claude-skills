#!/bin/bash
# AI MKP Models - CLI Tool
# Unified command-line interface for chart management, deployment, and Kubernetes operations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REGISTRY_URL="${REGISTRY_URL:-hub.fci.vn}"
REGISTRY_PROJECT="${REGISTRY_PROJECT:-ncp-modas/charts}"
NAMESPACE="${NAMESPACE:-llms}"
CHART_VERSION_BASE="2.0.0-patch-06"

# Get chart info from Chart.yaml
get_chart_name() {
  grep '^name:' "$CHART_DIR/Chart.yaml" | awk '{print $2}'
}

get_chart_version() {
  grep '^version:' "$CHART_DIR/Chart.yaml" | awk '{print $2}'
}

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
  echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${CYAN}║  AI MKP Models - Helm Chart Management                            ║${NC}"
  echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
  echo ""
}

print_success() {
  echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
  echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
  echo -e "${RED}✗ $1${NC}"
}

confirm_action() {
  local message="$1"
  print_warning "$message"
  read -p "Continue? [y/N] " -n 1 -r
  echo
  [[ $REPLY =~ ^[Yy]$ ]]
}

get_helmfile_for_env() {
  local env="$1"
  case "$env" in
    staging) echo "helmfile.yaml.gotmpl" ;;
    production-vn) echo "helmfile-production-vn.yaml.gotmpl" ;;
    production-jp) echo "helmfile-production-jp.yaml.gotmpl" ;;
    *) echo "helmfile.yaml.gotmpl" ;;
  esac
}

# ============================================================================
# Chart Commands
# ============================================================================

cmd_chart_package() {
  echo "=== Packaging chart ==="
  cd "$CHART_DIR"
  "$SCRIPT_DIR/build-package.sh"
}

cmd_chart_push() {
  local chart_name=$(get_chart_name)
  local chart_version=$(get_chart_version)
  local tgz_file="$CHART_DIR/dist/${chart_name}-${chart_version}.tgz"

  if [ ! -f "$tgz_file" ]; then
    print_error "Chart package not found: $tgz_file"
    echo "Run './scripts/cli.sh chart package' first"
    exit 1
  fi

  echo "=== Pushing chart to registry ==="
  echo "Chart: $chart_name:$chart_version"
  echo "Registry: oci://$REGISTRY_URL/$REGISTRY_PROJECT"

  helm push "$tgz_file" "oci://$REGISTRY_URL/$REGISTRY_PROJECT"
  print_success "Chart pushed successfully"
}

cmd_chart_verify() {
  echo "=== Verifying chart ==="
  cd "$CHART_DIR"

  echo "Linting chart..."
  helm lint .
  print_success "Lint passed"

  echo ""
  echo "Building dependencies for template check..."
  # Package sub-charts to charts/ for template generation
  rm -rf charts/* 2>/dev/null || true
  for chart in subcharts/*/; do
    if [ -f "$chart/Chart.yaml" ]; then
      helm package "$chart" -d charts/ > /dev/null
    fi
  done
  # Extract sub-charts
  for tgz in charts/*.tgz; do
    if [ -f "$tgz" ]; then
      tar -xzf "$tgz" -C charts/ 2>/dev/null
      rm "$tgz"
    fi
  done

  echo "Generating templates..."
  helm template test-release . --debug > /dev/null
  print_success "Template generation successful"

  # Cleanup
  rm -rf charts/*
}

cmd_chart_clean() {
  echo "=== Cleaning up ==="
  cd "$CHART_DIR"
  rm -rf charts/*.tgz charts/*/
  rm -rf dist/
  rm -f Chart.lock helmfile.lock
  print_success "Cleanup complete"
}

cmd_chart_version() {
  local chart_name=$(get_chart_name)
  local chart_version=$(get_chart_version)

  echo "Chart: $chart_name"
  echo "Version: $chart_version"
  echo "Registry: oci://$REGISTRY_URL/$REGISTRY_PROJECT"
  echo ""
  echo "Available versions:"
  echo "  Staging:    ${CHART_VERSION_BASE}-staging"
  echo "  Production: ${CHART_VERSION_BASE}-production"
  echo ""

  if echo "$chart_version" | grep -q "staging"; then
    echo "Current environment: STAGING 🧪"
  elif echo "$chart_version" | grep -q "production"; then
    echo "Current environment: PRODUCTION 🚀"
  else
    echo "Current environment: UNKNOWN ⚠️"
  fi
}

cmd_chart_set_version() {
  local env="$1"
  local new_version

  case "$env" in
    staging)
      new_version="${CHART_VERSION_BASE}-staging"
      ;;
    production)
      new_version="${CHART_VERSION_BASE}-production"
      ;;
    *)
      print_error "Invalid environment: $env"
      echo "Usage: ./scripts/cli.sh chart set-version staging|production"
      exit 1
      ;;
  esac

  echo "Setting version to $new_version..."
  sed -i "s/^version:.*/version: $new_version/" "$CHART_DIR/Chart.yaml"
  print_success "Chart version updated to: $new_version"
}

cmd_chart_login() {
  echo "=== Logging in to registry ==="
  helm registry login "$REGISTRY_URL"
}

# ============================================================================
# Deploy Commands
# ============================================================================

cmd_deploy_sync() {
  local env="$1"
  local helmfile=$(get_helmfile_for_env "$env")

  if [[ "$env" == production-* ]]; then
    echo "Deploying to PRODUCTION: $env"
    kubectl config current-context
    if ! confirm_action "You are about to deploy to PRODUCTION"; then
      echo "Aborted"
      exit 0
    fi
  fi

  echo "=== Syncing releases for $env ==="
  cd "$CHART_DIR"
  helmfile -f "$helmfile" -e "$env" sync
}

cmd_deploy_diff() {
  local env="$1"
  local helmfile=$(get_helmfile_for_env "$env")

  echo "=== Showing diff for $env ==="
  cd "$CHART_DIR"
  helmfile -f "$helmfile" -e "$env" diff
}

cmd_deploy_status() {
  local env="$1"
  local helmfile=$(get_helmfile_for_env "$env")

  echo "=== Status for $env ==="
  cd "$CHART_DIR"
  helmfile -f "$helmfile" -e "$env" status
}

cmd_deploy_destroy() {
  local env="$1"
  local helmfile=$(get_helmfile_for_env "$env")

  if [[ "$env" == production-* ]]; then
    print_warning "WARNING: This will destroy all releases in $env!"
    if ! confirm_action "Are you REALLY sure?"; then
      echo "Aborted"
      exit 0
    fi
  fi

  echo "=== Destroying releases for $env ==="
  cd "$CHART_DIR"
  helmfile -f "$helmfile" -e "$env" destroy
}

cmd_deploy_list() {
  local env="$1"
  local helmfile=$(get_helmfile_for_env "$env")

  echo "=== Listing releases for $env ==="
  cd "$CHART_DIR"
  helmfile -f "$helmfile" -e "$env" list
}

# ============================================================================
# Model Commands
# ============================================================================

cmd_model_deploy() {
  local model="$1"
  local env="$2"

  if [ -z "$model" ]; then
    print_error "Model name not specified"
    echo "Usage: ./scripts/cli.sh model deploy <model-name> -e <environment>"
    exit 1
  fi

  echo "=== Deploying model: $model ==="
  cd "$CHART_DIR"
  helmfile -e "$env" -l "name=$model" sync
}

cmd_model_destroy() {
  local model="$1"
  local env="$2"

  if [ -z "$model" ]; then
    print_error "Model name not specified"
    echo "Usage: ./scripts/cli.sh model destroy <model-name> -e <environment>"
    exit 1
  fi

  echo "=== Destroying model: $model ==="
  cd "$CHART_DIR"
  helmfile -e "$env" -l "name=$model" destroy
}

# ============================================================================
# Kubernetes Commands
# ============================================================================

cmd_k8s_status() {
  echo "=== Pods ==="
  kubectl get pods -n "$NAMESPACE"
  echo ""
  echo "=== Services ==="
  kubectl get svc -n "$NAMESPACE"
  echo ""
  echo "=== ScaledObjects ==="
  kubectl get scaledobjects -n "$NAMESPACE" 2>/dev/null || echo "KEDA not installed or no ScaledObjects"
}

cmd_k8s_logs() {
  local name="$1"

  if [ -z "$name" ]; then
    print_error "Release name not specified"
    echo "Usage: ./scripts/cli.sh k8s logs <release-name>"
    exit 1
  fi

  kubectl logs -n "$NAMESPACE" -l "app.kubernetes.io/instance=$name" -f --tail=100
}

cmd_k8s_describe() {
  local name="$1"

  if [ -z "$name" ]; then
    print_error "Release name not specified"
    echo "Usage: ./scripts/cli.sh k8s describe <release-name>"
    exit 1
  fi

  kubectl describe pod -n "$NAMESPACE" -l "app.kubernetes.io/instance=$name"
}

# ============================================================================
# Build-Push Shortcuts
# ============================================================================

cmd_build_push() {
  echo "=== Build and Push Chart ==="

  # Step 1: Package chart using build-package.sh
  cmd_chart_package

  # Step 2: Push to OCI registry
  cmd_chart_push

  echo ""
  echo "=========================================="
  print_success "Successfully built and pushed chart!"
  echo "=========================================="
}

# ============================================================================
# Help
# ============================================================================

show_help() {
  print_header
  echo "Usage: ./scripts/cli.sh <command> [subcommand] [options]"
  echo ""
  echo -e "${CYAN}📦 Chart Commands:${NC}"
  echo "  chart package              Package chart with sub-charts"
  echo "  chart push                 Push chart to OCI registry"
  echo "  chart verify               Lint + template check"
  echo "  chart clean                Clean up generated files"
  echo "  chart version              Show current version"
  echo "  chart set-version <env>    Set version (staging|production)"
  echo "  chart login                Login to OCI registry"
  echo ""
  echo -e "${CYAN}🚀 Deploy Commands:${NC}"
  echo "  deploy sync -e <env>       Sync releases"
  echo "  deploy diff -e <env>       Preview changes"
  echo "  deploy status -e <env>     Check status"
  echo "  deploy destroy -e <env>    Destroy releases"
  echo "  deploy list -e <env>       List releases"
  echo ""
  echo -e "${CYAN}🎯 Model Commands:${NC}"
  echo "  model deploy <name> -e <env>   Deploy specific model"
  echo "  model destroy <name> -e <env>  Remove specific model"
  echo ""
  echo -e "${CYAN}🔍 Kubernetes Commands:${NC}"
  echo "  k8s status                 Show pods, services, scaledobjects"
  echo "  k8s logs <name>            View logs for release"
  echo "  k8s describe <name>        Describe pods"
  echo ""
  echo -e "${CYAN}⚡ Quick Commands:${NC}"
  echo "  build-push                 Build and push chart (version from Chart.yaml)"
  echo ""
  echo "Environments: staging, production-vn, production-jp"
  echo ""
}

# ============================================================================
# Main
# ============================================================================

parse_env_flag() {
  local env="staging"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -e|--env)
        env="$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  echo "$env"
}

main() {
  local cmd="$1"
  local subcmd="$2"
  shift 2 2>/dev/null || true

  case "$cmd" in
    chart)
      case "$subcmd" in
        package) cmd_chart_package ;;
        push) cmd_chart_push ;;
        verify) cmd_chart_verify ;;
        clean) cmd_chart_clean ;;
        version) cmd_chart_version ;;
        set-version) cmd_chart_set_version "$1" ;;
        login) cmd_chart_login ;;
        *) show_help; exit 1 ;;
      esac
      ;;
    deploy)
      local env=$(parse_env_flag "$@")
      case "$subcmd" in
        sync) cmd_deploy_sync "$env" ;;
        diff) cmd_deploy_diff "$env" ;;
        status) cmd_deploy_status "$env" ;;
        destroy) cmd_deploy_destroy "$env" ;;
        list) cmd_deploy_list "$env" ;;
        *) show_help; exit 1 ;;
      esac
      ;;
    model)
      local model="$1"
      shift 2>/dev/null || true
      local env=$(parse_env_flag "$@")
      case "$subcmd" in
        deploy) cmd_model_deploy "$model" "$env" ;;
        destroy) cmd_model_destroy "$model" "$env" ;;
        *) show_help; exit 1 ;;
      esac
      ;;
    k8s)
      case "$subcmd" in
        status) cmd_k8s_status ;;
        logs) cmd_k8s_logs "$1" ;;
        describe) cmd_k8s_describe "$1" ;;
        *) show_help; exit 1 ;;
      esac
      ;;
    build-push)
      cmd_build_push
      ;;
    help|--help|-h|"")
      show_help
      ;;
    *)
      print_error "Unknown command: $cmd"
      show_help
      exit 1
      ;;
  esac
}

main "$@"
