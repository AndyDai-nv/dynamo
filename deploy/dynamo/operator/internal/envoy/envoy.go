/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 Atalaya Tech. Inc
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
 */

package envoy

import (
	"bytes"
	"strings"
	"text/template"
)

const (
	EnvoyAdminPort = 9901
)

type CreateEnvoyConfig struct {
	ListenPort              int
	DebugHeaderName         string
	DebugHeaderValue        string
	DebugServerAddress      string
	DebugServerPort         int
	ProductionServerAddress string
	ProductionServerPort    int
}

const configTemplate = `
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 0.0.0.0
        port_value: {{ .Config.ListenPort }}

    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http
          access_log:
          - name: envoy.access_loggers.stdout
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.access_loggers.stream.v3.StdoutAccessLog
          http_filters:
          - name: envoy.filters.http.router
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
          route_config:
            name: local_route
            virtual_hosts:
            - name: backend
              domains: ["*"]
              routes:
              - match:
                  prefix: "/"
                  headers:
                    - name: "{{ .Config.DebugHeaderName }}"
                      exact_match: "{{ .Config.DebugHeaderValue }}"
                route:
                  cluster: service_debug
              - match:
                  prefix: "/"
                route:
                  cluster: service_production

  clusters:
  - name: service_debug
    connect_timeout: 0.25s
    type: strict_dns
    dns_lookup_family: v4_only
    lb_policy: round_robin
    load_assignment:
      cluster_name: service_debug
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: {{ .Config.DebugServerAddress }}
                port_value: {{ .Config.DebugServerPort }}

  - name: service_production
    connect_timeout: 0.25s
    type: strict_dns
    dns_lookup_family: v4_only
    lb_policy: round_robin
    load_assignment:
      cluster_name: service_production
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: {{ .Config.ProductionServerAddress }}
                port_value: {{ .Config.ProductionServerPort }}

admin:
    access_log_path: /dev/null
    address:
        socket_address:
            address: 0.0.0.0
            port_value: {{ .AdminPort }}
`

func GenerateEnvoyConfigurationContent(config CreateEnvoyConfig) (string, error) {
	t := template.Must(template.New("envoy").Parse(configTemplate))
	buf := new(bytes.Buffer)
	err := t.Execute(buf, map[string]interface{}{
		"Config":    config,
		"AdminPort": EnvoyAdminPort,
	})
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(buf.String()), nil
}
