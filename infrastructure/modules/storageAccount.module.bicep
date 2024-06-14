@description('Name for the storage account')
param storageAccountName string
@description('Location for resource.')
param location string
@description('What language was used to deploy this resource')
param language string
@description('User Assigned Identity')
param uidName string

@description('Storage Account type')
@allowed([
  'Premium_LRS'
  'Premium_ZRS'
  'Standard_GRS'
  'Standard_GZRS'
  'Standard_LRS'
  'Standard_RAGRS'
  'Standard_RAGZRS'
  'Standard_ZRS'
])
param storageAccountType string = 'Standard_LRS'
param keyVaultName string

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource uid 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-07-31-preview' existing = {
  name: uidName
}

@description('This is the built-in ACR Pull RBAC role. See https://docs.microsoft.com/azure/role-based-access-control/built-in-roles#contributor')
resource blobContributorRBAC 'Microsoft.Authorization/roleDefinitions@2022-05-01-preview' existing = {
  scope: subscription()
  name: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
}

resource sa 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: toLower(replace('sa${storageAccountName}','-',''))
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: storageAccountType
  }
  tags:{
    Language: language
  }
  kind: 'StorageV2'
  properties: {}
}

resource storageAccountBlobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  name: 'default'
  parent: sa
}

resource storageAccountContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: 'rawdata'
  parent: storageAccountBlobService
}

resource saBlobContributorRBAC 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(sa.id, uid.id, blobContributorRBAC.id)
  scope: sa
  properties: {
    roleDefinitionId: blobContributorRBAC.id
    principalId: uid.properties.principalId
    principalType: 'ServicePrincipal'
  }

}
resource storageAccountEndpoint 'Microsoft.KeyVault/vaults/secrets@2023-07-01'= {
  parent : keyVault
  name: 'sa-endpoint'
  properties: {
    value:  sa.properties.primaryEndpoints.blob
  }
}
