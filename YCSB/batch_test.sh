#!/bin/bash

# YCSB批量测试脚本
# 作者: 自动生成
# 用途: 执行一系列YCSB测试，支持不同workload和properties配置

# ===========================================
# 全局配置变量
# ===========================================

# 默认配置
DEFAULT_YCSB_PATH="./ycsb"
DEFAULT_DB_TYPE="rocksdb"
DEFAULT_PROPERTIES_FILE="rocksdb/rocksdb.properties"
DEFAULT_THREAD_COUNT=8

# 数据库和NVM路径配置
DB_PATH="/mnt/nvme0n1/guoteng/walsmtest/tmp/db_nvm_l0"
NVM_PATH="/mnt/pmem0.7/guoteng/nodememory"

# 日志配置
LOG_DIR="./test_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ===========================================
# 工具函数
# ===========================================

# 打印带时间戳的日志信息
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

# 打印错误信息
log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# 打印警告信息
log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1"
}

# ===========================================
# 资源清理函数
# ===========================================

# 清理数据库和NVM存储资源
cleanup_resources() {
    log_info "开始清理资源..."
    
    local cleanup_success=true
    
    # 清理数据库目录
    if [ -d "${DB_PATH}" ]; then
        log_info "清理数据库目录: ${DB_PATH}"
        if [ "$(ls -A "${DB_PATH}" 2>/dev/null)" ]; then
            rm -rf "${DB_PATH}"/*
            if [ $? -eq 0 ]; then
                log_info "数据库目录清理完成"
            else
                log_error "数据库目录清理失败"
                cleanup_success=false
            fi
        else
            log_info "数据库目录为空，无需清理"
        fi
    else
        log_info "数据库目录不存在: ${DB_PATH} (正常情况)"
    fi
    
    # 清理NVM路径
    if [ -f "${NVM_PATH}" ] || [ -d "${NVM_PATH}" ]; then
        log_info "清理NVM路径: ${NVM_PATH}"
        rm -rf "${NVM_PATH}"
        if [ $? -eq 0 ]; then
            log_info "NVM路径清理完成"
        else
            log_error "NVM路径清理失败"
            cleanup_success=false
        fi
    else
        log_info "NVM路径不存在: ${NVM_PATH} (正常情况)"
    fi
    
    if [ "$cleanup_success" = true ]; then
        log_info "资源清理完成"
        return 0
    else
        log_error "资源清理过程中出现错误"
        return 1
    fi
}

# ===========================================
# YCSB测试执行函数
# ===========================================

# 执行单个YCSB测试
run_single_ycsb_test() {
    local workload="$1"
    local properties_file="$2"
    local additional_params="$3"
    local test_name="$4"
    
    log_info "开始执行YCSB测试: ${test_name}"
    log_info "  - Workload: ${workload}"
    log_info "  - Properties: ${properties_file}"
    log_info "  - Thread Count: ${DEFAULT_THREAD_COUNT}"
    
    # 检查workload文件是否存在
    if [ ! -f "${workload}" ]; then
        log_error "Workload文件不存在: ${workload}"
        return 1
    fi
    
    # 检查properties文件是否存在
    if [ ! -f "${properties_file}" ]; then
        log_error "Properties文件不存在: ${properties_file}"
        return 1
    fi
    
    # 创建日志目录
    mkdir -p "${LOG_DIR}"
    
    # 生成日志文件名
    local log_file="${LOG_DIR}/${test_name}_${TIMESTAMP}.log"
    
    # 构建YCSB命令
    local ycsb_cmd="${DEFAULT_YCSB_PATH} -load -run -db ${DEFAULT_DB_TYPE} -P ${workload} -P ${properties_file} -p threadcount=${DEFAULT_THREAD_COUNT}"
    
    # 添加额外参数
    if [ -n "${additional_params}" ]; then
        ycsb_cmd="${ycsb_cmd} ${additional_params}"
    fi
    
    # 添加统计输出
    ycsb_cmd="${ycsb_cmd} -s"
    
    log_info "执行命令: ${ycsb_cmd}"
    
    # 执行YCSB测试并记录日志
    echo "开始时间: $(date)" > "${log_file}"
    echo "命令: ${ycsb_cmd}" >> "${log_file}"
    echo "======================================" >> "${log_file}"
    
    eval "${ycsb_cmd}" 2>&1 | tee -a "${log_file}"
    local exit_code=${PIPESTATUS[0]}
    
    echo "======================================" >> "${log_file}"
    echo "结束时间: $(date)" >> "${log_file}"
    echo "退出码: ${exit_code}" >> "${log_file}"
    
    if [ ${exit_code} -eq 0 ]; then
        log_info "测试完成: ${test_name} (日志: ${log_file})"
        return 0
    else
        log_error "测试失败: ${test_name} (退出码: ${exit_code})"
        return 1
    fi
}

# ===========================================
# 批量测试函数
# ===========================================

# 执行批量测试
run_batch_tests() {
    local config_file="$1"
    
    if [ ! -f "${config_file}" ]; then
        log_error "配置文件不存在: ${config_file}"
        return 1
    fi
    
    log_info "开始批量测试，配置文件: ${config_file}"
    
    local test_count=0
    local success_count=0
    local failed_tests=()
    
    # 读取配置文件并执行测试
    while IFS='|' read -r test_name workload properties_file additional_params; do
        # 跳过注释行和空行
        [[ ${test_name} =~ ^#.*$ ]] && continue
        [[ -z ${test_name} ]] && continue
        
        test_count=$((test_count + 1))
        
        log_info "执行测试 ${test_count}: ${test_name}"
        
        # 清理资源
        cleanup_resources
        if [ $? -ne 0 ]; then
            log_error "资源清理失败，跳过测试: ${test_name}"
            failed_tests+=("${test_name}")
            continue
        fi
        
        # 确定properties文件路径
        local actual_properties_file
        if [ -n "${properties_file}" ] && [ "${properties_file}" != " " ]; then
            actual_properties_file="${properties_file}"
        else
            actual_properties_file="${DEFAULT_PROPERTIES_FILE}"
        fi
        
        # 执行测试
        run_single_ycsb_test "${workload}" "${actual_properties_file}" "${additional_params}" "${test_name}"
        if [ $? -eq 0 ]; then
            success_count=$((success_count + 1))
            log_info "测试成功: ${test_name}"
        else
            failed_tests+=("${test_name}")
            log_error "测试失败: ${test_name}"
        fi
        
        log_info "测试 ${test_count} 完成: ${test_name}"
        echo "----------------------------------------"
        
    done < "${config_file}"
    
    # 输出测试结果汇总
    log_info "批量测试完成"
    log_info "总测试数: ${test_count}"
    log_info "成功测试数: ${success_count}"
    log_info "失败测试数: $((test_count - success_count))"
    
    if [ ${#failed_tests[@]} -gt 0 ]; then
        log_warn "失败的测试:"
        for failed_test in "${failed_tests[@]}"; do
            log_warn "  - ${failed_test}"
        done
    fi
    
    return 0
}

# ===========================================
# 主函数
# ===========================================

# 显示帮助信息
show_help() {
    cat << EOF
YCSB批量测试脚本使用说明:

用法:
    $0 [选项] [参数]

选项:
    -h, --help              显示此帮助信息
    -c, --cleanup           仅执行资源清理
    -s, --single            执行单个测试
    -b, --batch             执行批量测试
    -l, --list              列出可用的workload文件

单个测试参数:
    -w, --workload          指定workload文件 (必需)
    -P, --properties        指定properties文件 (可选，默认: ${DEFAULT_PROPERTIES_FILE})
    -p, --params            额外参数 (可选)
    -n, --name              测试名称 (可选)

批量测试参数:
    -f, --config-file       批量测试配置文件 (必需)

配置文件格式 (使用|分隔):
    测试名称|workload路径|properties文件路径(可选)|额外参数(可选)

示例:
    # 执行单个测试
    $0 -s -w workloads/workloada -n test1

    # 执行单个测试并指定properties文件
    $0 -s -w workloads/workloada -P custom.properties -n test1

    # 执行批量测试
    $0 -b -f batch_config.txt

    # 仅清理资源
    $0 -c

EOF
}

# 列出可用的workload文件
list_workloads() {
    log_info "可用的workload文件:"
    find workloads/ -name "workload*" -type f | sort | while read -r workload; do
        echo "  - ${workload}"
    done
}

# 主函数
main() {
    local action=""
    local workload=""
    local properties_file=""
    local additional_params=""
    local test_name=""
    local config_file=""
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--cleanup)
                action="cleanup"
                shift
                ;;
            -s|--single)
                action="single"
                shift
                ;;
            -b|--batch)
                action="batch"
                shift
                ;;
            -l|--list)
                list_workloads
                exit 0
                ;;
            -w|--workload)
                workload="$2"
                shift 2
                ;;
            -P|--properties)
                properties_file="$2"
                shift 2
                ;;
            -p|--params)
                additional_params="$2"
                shift 2
                ;;
            -n|--name)
                test_name="$2"
                shift 2
                ;;
            -f|--config-file)
                config_file="$2"
                shift 2
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查YCSB可执行文件
    if [ ! -f "${DEFAULT_YCSB_PATH}" ]; then
        log_error "YCSB可执行文件不存在: ${DEFAULT_YCSB_PATH}"
        exit 1
    fi
    
    # 根据action执行相应操作
    case "${action}" in
        cleanup)
            cleanup_resources
            ;;
        single)
            if [ -z "${workload}" ]; then
                log_error "执行单个测试需要指定workload文件"
                show_help
                exit 1
            fi
            
            if [ -z "${test_name}" ]; then
                test_name="single_test_$(basename ${workload})"
            fi
            
            # 确定properties文件路径
            if [ -z "${properties_file}" ]; then
                properties_file="${DEFAULT_PROPERTIES_FILE}"
            fi
            
            # 清理资源
            cleanup_resources
            
            # 执行测试
            run_single_ycsb_test "${workload}" "${properties_file}" "${additional_params}" "${test_name}"
            local result=$?
            
            exit ${result}
            ;;
        batch)
            if [ -z "${config_file}" ]; then
                log_error "执行批量测试需要指定配置文件"
                show_help
                exit 1
            fi
            
            run_batch_tests "${config_file}"
            ;;
        *)
            log_error "需要指定操作类型 (-c, -s, -b, -l)"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
