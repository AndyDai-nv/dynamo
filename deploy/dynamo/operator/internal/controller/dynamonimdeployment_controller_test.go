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

package controller

import (
	"context"
	"fmt"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIsDeploymentReady(t *testing.T) {
	type args struct {
		deployment *appsv1.Deployment
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "deployment is nil",
			args: args{
				deployment: nil,
			},
			want: false,
		},
		{
			name: "not ready",
			args: args{
				deployment: &appsv1.Deployment{
					Spec: appsv1.DeploymentSpec{},
					Status: appsv1.DeploymentStatus{
						Conditions: []appsv1.DeploymentCondition{
							{
								Type:   appsv1.DeploymentAvailable,
								Status: corev1.ConditionFalse,
							},
						},
					},
				},
			},
			want: false,
		},
		{
			name: "not ready (paused)",
			args: args{
				deployment: &appsv1.Deployment{
					Spec: appsv1.DeploymentSpec{
						Paused: true,
					},
				},
			},
			want: false,
		},
		{
			name: "ready",
			args: args{
				deployment: &appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{
						Generation: 1,
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: &[]int32{1}[0],
					},
					Status: appsv1.DeploymentStatus{
						ObservedGeneration: 1,
						UpdatedReplicas:    1,
						AvailableReplicas:  1,
						Conditions: []appsv1.DeploymentCondition{
							{
								Type:   appsv1.DeploymentAvailable,
								Status: corev1.ConditionTrue,
							},
						},
					},
				},
			},
			want: true,
		},
		{
			name: "ready (no desired replicas)",
			args: args{
				deployment: &appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{
						Generation: 1,
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: &[]int32{0}[0],
					},
				},
			},
			want: true,
		},
		{
			name: "not ready (condition false)",
			args: args{
				deployment: &appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{
						Generation: 1,
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: &[]int32{1}[0],
					},
					Status: appsv1.DeploymentStatus{
						ObservedGeneration: 1,
						UpdatedReplicas:    1,
						AvailableReplicas:  1,
						Conditions: []appsv1.DeploymentCondition{
							{
								Type:   appsv1.DeploymentAvailable,
								Status: corev1.ConditionFalse,
							},
						},
					},
				},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsDeploymentReady(tt.args.deployment); got != tt.want {
				t.Errorf("IsDeploymentReady() = %v, want %v", got, tt.want)
			}
		})
	}
}

type mockEtcdStorage struct {
	deleteKeysFunc func(ctx context.Context, prefix string) error
}

func (m *mockEtcdStorage) DeleteKeys(ctx context.Context, prefix string) error {
	return m.deleteKeysFunc(ctx, prefix)
}

func TestDynamoNimDeploymentReconciler_FinalizeResource(t *testing.T) {
	type fields struct {
		EtcdStorage etcdStorage
	}
	type args struct {
		ctx                 context.Context
		dynamoNimDeployment *v1alpha1.DynamoNimDeployment
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		{
			name: "delete etcd keys",
			fields: fields{
				EtcdStorage: &mockEtcdStorage{
					deleteKeysFunc: func(ctx context.Context, prefix string) error {
						if prefix == "/default/components/service1" {
							return nil
						}
						return fmt.Errorf("invalid prefix: %s", prefix)
					},
				},
			},
			args: args{
				ctx: context.Background(),
				dynamoNimDeployment: &v1alpha1.DynamoNimDeployment{
					Spec: v1alpha1.DynamoNimDeploymentSpec{
						DynamoNimDeploymentSharedSpec: v1alpha1.DynamoNimDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default"}[0],
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "delete etcd keys (error)",
			fields: fields{
				EtcdStorage: &mockEtcdStorage{
					deleteKeysFunc: func(ctx context.Context, prefix string) error {
						return fmt.Errorf("invalid prefix: %s", prefix)
					},
				},
			},
			args: args{
				ctx: context.Background(),
				dynamoNimDeployment: &v1alpha1.DynamoNimDeployment{
					Spec: v1alpha1.DynamoNimDeploymentSpec{
						DynamoNimDeploymentSharedSpec: v1alpha1.DynamoNimDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default"}[0],
						},
					},
				},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &DynamoNimDeploymentReconciler{
				EtcdStorage: tt.fields.EtcdStorage,
			}
			if err := r.FinalizeResource(tt.args.ctx, tt.args.dynamoNimDeployment); (err != nil) != tt.wantErr {
				t.Errorf("DynamoNimDeploymentReconciler.FinalizeResource() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
