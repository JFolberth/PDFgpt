@description('Name for the Form Recognizer resource.')
param aciName string
@description('Image Name')
param aciImage string
@description('Image Name and Tag')
param aciImageNameTag string
@description('SKU for Azure Container Instance.')
@allowed(['Standard', 'Confidental', 'Dedicated'])
param aciSKU string = 'Standard'
@description('Location for resource.')
param imagecpu int = 2
param imagemem string = '1.5'
param location string
@description('What language was used to deploy this resource')
param language string
@description('What OS type was used to deploy this resource')
@allowed(['Linux', 'Windows'])
param osType string = 'Linux'
@description('DNS Label Name')
param dnsLabel string
@description('ACR Name')
param acrName string
@description('User Assigned Identity')
param uidName string
@description('ACR Admin Password')
@secure()
param acrAdminPassword string
@description('ACR Admin Password')
@secure()
param acrUserName string


resource uid 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: uidName
}

var userAssignedIdentity = {
  Default:{
    '${uid.id}' : {}
  }
}

@description('Name of KeyVault to store values in')
param keyVaultName string

resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' existing = {
  name: keyVaultName
}

@description('This is the built-in ACR Pull RBAC role. See https://docs.microsoft.com/azure/role-based-access-control/built-in-roles#contributor')
resource acrPullRBAC 'Microsoft.Authorization/roleDefinitions@2018-01-01-preview' existing = {
  scope: subscription()
  name: '7f951dda-4ed3-4680-a7ca-43fe172d538d'
}

@description('This is the built-in Key Vault Seceret RBAC role. See https://docs.microsoft.com/azure/role-based-access-control/built-in-roles#contributor')
resource keyVaultSecretUserRBAC 'Microsoft.Authorization/roleDefinitions@2018-01-01-preview' existing = {
  scope: subscription()
  name: '4633458b-17de-408a-b874-0445c86b69e6'
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' existing = {
  name: acrName
}

resource aci 'Microsoft.ContainerInstance/containerGroups@2023-05-01' = {
  name: 'aci-${aciName}'
  location: location
  tags: {
    language: language
  }
 
 identity: {
    type: 'UserAssigned'
    userAssignedIdentities: userAssignedIdentity['Default']
  }
  properties: {
    sku: aciSKU
    containers: [
     
      {
      name: aciImage
      properties:{
        image: aciImageNameTag
        ports: [
          {
            port: 80
            protocol: 'TCP'
          }
          {
            port: 443
            protocol: 'TCP'
          }
        ]
        resources: {
          requests: {
            cpu: imagecpu
            memoryInGB: imagemem
          }
        }
      }
      
      }
    ]
    ipAddress: {
      ports: [
        {
          port: 80
          protocol: 'TCP'
        }
        {
          port: 443
          protocol: 'TCP'
        }
      ]
      type: 'Public'
      dnsNameLabel: dnsLabel
    }
    osType: osType
    imageRegistryCredentials: [
      {
        server: acr.properties.loginServer
        username: acrUserName
        password: acrAdminPassword
      }
    ]

  }
  dependsOn: [
    aciPullRBAC
  ]
}

resource aciPullRBAC 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, uid.id, acrPullRBAC.id)
  scope: acr
  properties: {
    roleDefinitionId: acrPullRBAC.id
    principalId: uid.properties.principalId
    principalType: 'ServicePrincipal'
  }

}


resource keyVaultUserRBAC 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, uid.id, keyVaultSecretUserRBAC.id)
  scope: keyVault
  properties: {
    roleDefinitionId: keyVaultSecretUserRBAC.id
    principalId: uid.properties.principalId
    principalType: 'ServicePrincipal'
  }
  dependsOn: [
    uid, keyVault
  ]
}

